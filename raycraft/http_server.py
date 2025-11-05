"""
RayCraft HTTP Server - Gym环境的HTTP包装器

这个服务器提供RESTful API来访问Ray分布式环境池。
设计原则：简单、直接、没有过度设计。
"""

import io
import uuid
import base64
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import ray
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from raycraft.ray.global_pool import get_global_env_pool


# ============================================================================
# 数据模型 - 越简单越好
# ============================================================================

class CreateEnvRequest(BaseModel):
    env_name: str = Field(..., description="环境名称")
    env_kwargs: Dict[str, Any] = Field(default_factory=dict, description="环境参数")

class CreateEnvResponse(BaseModel):
    env_id: str
    observation_space: Dict[str, Any]
    action_space: Dict[str, Any]

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None

class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any]

class StepRequest(BaseModel):
    action: Any

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
# 工具函数 - 只做必要的事
# ============================================================================

def compress_image(image: np.ndarray, quality: int = 85) -> str:
    """
    压缩图像到JPEG格式并转为base64

    Args:
        image: numpy数组 (H, W, 3) uint8
        quality: JPEG质量 (1-100)

    Returns:
        base64编码的JPEG字符串
    """
    # 简单粗暴：numpy -> PIL -> JPEG -> base64
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    jpeg_bytes = buffer.getvalue()
    return base64.b64encode(jpeg_bytes).decode('utf-8')


def decompress_image(b64_str: str) -> np.ndarray:
    """
    从base64解压JPEG到numpy数组

    Args:
        b64_str: base64编码的JPEG字符串

    Returns:
        numpy数组 (H, W, 3) uint8
    """
    jpeg_bytes = base64.b64decode(b64_str)
    buffer = io.BytesIO(jpeg_bytes)
    pil_image = Image.open(buffer)
    return np.array(pil_image)


def serialize_observation(obs: Dict[str, Any], compress: bool = True) -> Dict[str, Any]:
    """
    序列化观察数据

    只处理 'rgb' 键的图像压缩，其他数据原样返回
    """
    result = {}
    for key, value in obs.items():
        if key == 'rgb' and compress and isinstance(value, np.ndarray):
            # 压缩RGB图像
            result[key] = {
                'type': 'jpeg',
                'data': compress_image(value)
            }
        elif isinstance(value, np.ndarray):
            # 其他numpy数组转列表
            result[key] = value.tolist()
        else:
            # 原样返回
            result[key] = value
    return result


def deserialize_action(action: Any) -> Any:
    """
    反序列化动作数据

    如果是列表，转为numpy数组；否则原样返回
    """
    if isinstance(action, list):
        return np.array(action)
    return action


# ============================================================================
# FastAPI应用
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - 启动和关闭Ray"""
    # 启动
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # 确保全局环境池存在
    get_global_env_pool()

    yield

    # 关闭
    # Ray会自动清理，不需要显式关闭


app = FastAPI(
    title="RayCraft HTTP API",
    description="Minecraft环境的HTTP接口",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# API路由 - 越简单越好
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


@app.post("/envs", response_model=CreateEnvResponse, status_code=status.HTTP_201_CREATED)
async def create_environment(request: CreateEnvRequest):
    """创建新环境"""
    try:
        env_pool = get_global_env_pool()

        # 生成UUID
        env_id = str(uuid.uuid4())

        # 创建环境
        env_ref = ray.get(env_pool.create_env.remote(
            env_id=env_id,
            env_name=request.env_name,
            env_kwargs=request.env_kwargs
        ))

        # 获取空间信息
        obs_space = ray.get(env_ref.observation_space.remote())
        act_space = ray.get(env_ref.action_space.remote())

        return CreateEnvResponse(
            env_id=env_id,
            observation_space={'type': str(type(obs_space).__name__)},
            action_space={'type': str(type(act_space).__name__)}
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create environment: {str(e)}"
        )


@app.post("/envs/{env_id}/reset", response_model=ResetResponse)
async def reset_environment(env_id: str, request: ResetRequest):
    """重置环境"""
    try:
        env_pool = get_global_env_pool()
        env_ref = ray.get(env_pool.get_env.remote(env_id))

        if env_ref is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        # 准备reset参数
        reset_kwargs = {}
        if request.seed is not None:
            reset_kwargs['seed'] = request.seed
        if request.options is not None:
            reset_kwargs['options'] = request.options

        # 执行reset
        obs, info = ray.get(env_ref.reset.remote(**reset_kwargs))

        return ResetResponse(
            observation=serialize_observation(obs),
            info=info
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset environment: {str(e)}"
        )


@app.post("/envs/{env_id}/step", response_model=StepResponse)
async def step_environment(env_id: str, request: StepRequest):
    """执行环境步进"""
    try:
        env_pool = get_global_env_pool()
        env_ref = ray.get(env_pool.get_env.remote(env_id))

        if env_ref is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        # 反序列化动作
        action = deserialize_action(request.action)

        # 执行step
        obs, reward, terminated, truncated, info = ray.get(
            env_ref.step.remote(action)
        )

        return StepResponse(
            observation=serialize_observation(obs),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to step environment: {str(e)}"
        )


@app.delete("/envs/{env_id}", status_code=status.HTTP_204_NO_CONTENT)
async def close_environment(env_id: str):
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
            detail=f"Failed to close environment: {str(e)}"
        )


@app.get("/envs", response_model=List[str])
async def list_environments():
    """列出所有环境ID"""
    try:
        env_pool = get_global_env_pool()
        env_ids = ray.get(env_pool.list_envs.remote())
        return env_ids

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list environments: {str(e)}"
        )


# ============================================================================
# 错误处理
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
