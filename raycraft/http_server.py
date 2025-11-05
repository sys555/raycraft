"""
RayCraft HTTP Server

依赖：
- fastapi
- uvicorn
- ray
- pillow
"""

import uuid
import base64
import io
from contextlib import asynccontextmanager
from typing import Dict, Any

import ray
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from raycraft.ray.global_pool import get_global_env_pool


# ============================================================================
# 数据模型
# ============================================================================

class BatchCreateRequest(BaseModel):
    count: int = Field(..., ge=1, description="环境数量，必须>=1")
    env_name: str = Field(default="minecraft", description="环境名称")
    env_kwargs: Dict[str, Any] = Field(default_factory=dict, description="环境参数")


class BatchCreateResponse(BaseModel):
    env_ids: list


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any]


class StepRequest(BaseModel):
    action: str = Field(..., description="JSON格式的action字符串")


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    ray_initialized: bool
    num_environments: int


# ============================================================================
# 辅助函数
# ============================================================================

def serialize_value(value: Any) -> Any:
    """递归序列化值，处理numpy数组和Ray对象"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    elif 'ray' in str(type(value).__module__):
        # Ray对象（如ActorID, ObjectRef等）转为字符串
        return str(value)
    else:
        return value


def serialize_observation(obs: Dict[str, Any]) -> Dict[str, Any]:
    """序列化observation，压缩RGB图像"""
    result = {}
    for key, value in obs.items():
        if key == 'rgb' and isinstance(value, np.ndarray):
            # 压缩为JPEG
            pil_image = Image.fromarray(value)
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=85)
            jpeg_bytes = buffer.getvalue()
            result[key] = {
                'type': 'jpeg',
                'data': base64.b64encode(jpeg_bytes).decode('utf-8')
            }
        elif isinstance(value, np.ndarray):
            # 其他numpy数组转list
            result[key] = value.tolist()
        else:
            result[key] = serialize_value(value)
    return result


# ============================================================================
# FastAPI 应用
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化Ray"""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    get_global_env_pool()  # 确保命名actor存在
    yield


app = FastAPI(
    title="RayCraft HTTP API",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# API 路由
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    try:
        env_pool = get_global_env_pool()
        num_envs = ray.get(env_pool.get_num_envs.remote())
        return HealthResponse(
            status="healthy",
            ray_initialized=ray.is_initialized(),
            num_environments=num_envs
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/batch/envs", response_model=BatchCreateResponse)
async def batch_create_envs(request: BatchCreateRequest):
    """批量创建环境"""
    try:
        env_pool = get_global_env_pool()

        # 生成UUIDs
        env_ids = [str(uuid.uuid4()) for _ in range(request.count)]
        configs = [request.env_kwargs] * request.count

        # 并行创建
        success = ray.get(env_pool.create_envs.remote(env_ids, configs))

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create some environments"
            )

        return BatchCreateResponse(env_ids=env_ids)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create environments: {str(e)}"
        )


@app.post("/envs/{env_id}/reset", response_model=ResetResponse)
async def reset_env(env_id: str):
    """Reset环境"""
    try:
        env_pool = get_global_env_pool()
        env_ref = ray.get(env_pool.get_env.remote(env_id))

        if env_ref is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        result = ray.get(env_ref.reset.remote())

        # MCEnvActor.reset() 只返回 obs，不是 (obs, info)
        # 兼容处理
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        return ResetResponse(
            observation=serialize_observation(obs),
            info=serialize_value(info)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset: {str(e)}"
        )


@app.post("/envs/{env_id}/step", response_model=StepResponse)
async def step_env(env_id: str, request: StepRequest):
    """Step环境"""
    try:
        env_pool = get_global_env_pool()
        env_ref = ray.get(env_pool.get_env.remote(env_id))

        if env_ref is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        result = ray.get(env_ref.step.remote(request.action))

        # MCEnvActor.step() 返回旧版Gym格式 (obs, reward, done, info)
        # 需要转换为新版格式 (obs, reward, terminated, truncated, info)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False  # 旧版没有truncated，默认False
        else:
            obs, reward, terminated, truncated, info = result

        return StepResponse(
            observation=serialize_observation(obs),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=serialize_value(info)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to step: {str(e)}"
        )


@app.delete("/envs/{env_id}", status_code=status.HTTP_204_NO_CONTENT)
async def close_env(env_id: str):
    """关闭环境"""
    try:
        env_pool = get_global_env_pool()
        success = ray.get(env_pool.close_env.remote(env_id))

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        return None

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to close: {str(e)}"
        )


# ============================================================================
# 启动
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
