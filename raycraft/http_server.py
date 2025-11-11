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
import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union

import ray
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, status, Request
from pydantic import BaseModel, Field

from raycraft.ray.global_pool import get_global_env_pool


# ============================================================================
# 日志配置
# ============================================================================

def setup_logging():
    """配置日志系统：文件 + 控制台"""
    # 日志目录
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件路径（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"http_server_{timestamp}.log"

    # 配置root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除已有handlers
    logger.handlers.clear()

    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 文件handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"=" * 80)
    logger.info(f"HTTP Server logging initialized")
    logger.info(f"Log file: {log_file}")
    logger.info(f"=" * 80)

    return logger


# 初始化日志
logger = setup_logging()


# ============================================================================
# 数据模型
# ============================================================================

class BatchCreateRequest(BaseModel):
    count: int = Field(..., ge=1, description="环境数量，必须>=1")
    env_name: str = Field(default="minecraft", description="环境名称")
    env_kwargs: Union[Dict[str, Any], str] = Field(default_factory=dict, description="环境参数（dict）或YAML配置文件路径（str）")


class BatchCreateResponse(BaseModel):
    env_ids: list


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    info: Dict[str, Any]


class StepRequest(BaseModel):
    action: Union[str, Dict[str, int]] = Field(
        ...,
        description="Action格式支持两种：1) LLM格式的JSON字符串 '[{\"action\": \"forward\"}]' 2) Agent格式的dict {'buttons': 5, 'camera': 222}"
    )


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

def _prepare_env_config(base_config: Union[Dict[str, Any], str], env_id: str) -> Dict[str, Any]:
    """为单个环境准备独立的配置，确保record_path使用UUID子目录

    Args:
        base_config: 基础配置（dict或YAML路径）
        env_id: 环境UUID

    Returns:
        修改后的配置字典（sim_callbacks为实例化的对象）
    """
    import copy
    from pathlib import Path
    from raycraft.utils.sim_callbacks_loader import load_simulator_setup_from_yaml

    # 1. 如果是YAML路径，先加载并实例化callbacks
    if isinstance(base_config, str):
        sim_callbacks, env_overrides = load_simulator_setup_from_yaml(base_config)

        # 修改RecordCallback的record_path（已实例化的对象）
        for callback in sim_callbacks:
            if hasattr(callback, 'record_path'):
                base_path = Path(callback.record_path)
                # 在基础路径下创建UUID子目录（保持为Path对象）
                callback.record_path = base_path / env_id
                # 创建新的UUID子目录
                callback.record_path.mkdir(parents=True, exist_ok=True)

        # 重新组合config（sim_callbacks为实例化的对象）
        config = dict(env_overrides)
        config['sim_callbacks'] = sim_callbacks
        return config

    # 2. 如果是dict，深拷贝并修改
    config = copy.deepcopy(base_config)

    # 3. 修改sim_callbacks中RecordCallback的record_path（已实例化的对象）
    if 'sim_callbacks' in config and isinstance(config['sim_callbacks'], list):
        for callback in config['sim_callbacks']:
            if hasattr(callback, 'record_path'):
                base_path = Path(callback.record_path)
                # 保持为Path对象
                callback.record_path = base_path / env_id
                # 创建新的UUID子目录
                callback.record_path.mkdir(parents=True, exist_ok=True)

    return config


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
    """启动时初始化Ray，关闭时清理资源"""
    logger.info("=" * 80)
    logger.info("Starting RayCraft HTTP Server...")
    logger.info("=" * 80)

    try:
        if not ray.is_initialized():
            logger.info("Initializing Ray...")
            ray.init(ignore_reinit_error=True)
            logger.info(f"Ray initialized: {ray.is_initialized()}")
            logger.info(f"Ray cluster resources: {ray.cluster_resources()}")
        else:
            logger.info("Ray already initialized")

        logger.info("Getting global environment pool...")
        env_pool = get_global_env_pool()
        logger.info(f"Environment pool ready: {env_pool}")

        logger.info("=" * 80)
        logger.info("RayCraft HTTP Server started successfully")
        logger.info("Listening on http://0.0.0.0:8000")
        logger.info("=" * 80)

        yield

    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise
    finally:
        logger.info("=" * 80)
        logger.info("Shutting down RayCraft HTTP Server...")
        logger.info("=" * 80)


app = FastAPI(
    title="RayCraft HTTP API",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# 请求日志中间件
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有HTTP请求的日志"""
    start_time = time.time()

    # 记录请求开始
    logger.info(f"→ {request.method} {request.url.path} - Client: {request.client.host}")

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        # 记录响应
        logger.info(
            f"← {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Duration: {duration:.3f}s"
        )

        return response

    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"✗ {request.method} {request.url.path} - "
            f"ERROR: {str(e)} - "
            f"Duration: {duration:.3f}s",
            exc_info=True
        )
        raise


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
    logger.info(f"Batch creating {request.count} environments (env_name={request.env_name})")

    try:
        env_pool = get_global_env_pool()

        # 生成UUIDs
        env_ids = [str(uuid.uuid4()) for _ in range(request.count)]
        logger.info(f"Generated env_ids: {env_ids[:3]}{'...' if len(env_ids) > 3 else ''}")

        # 为每个环境创建独立的config（避免共享同一个record_path）
        logger.info("Preparing environment configs...")
        configs = []
        for env_id in env_ids:
            config = _prepare_env_config(request.env_kwargs, env_id)
            configs.append(config)

        # 并行创建
        logger.info("Creating environments in parallel...")
        create_start = time.time()
        success = ray.get(env_pool.create_envs.remote(env_ids, configs))
        create_duration = time.time() - create_start
        logger.info(f"Environment creation completed in {create_duration:.2f}s")

        if not success:
            logger.error("Failed to create some environments")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create some environments"
            )

        # 后台自动触发 reset（异步，不等待完成）
        # 客户端可通过 get_reset_result 获取结果
        logger.info("Triggering async reset for all environments...")
        for env_id in env_ids:
            env_ref = ray.get(env_pool.get_env.remote(env_id))
            if env_ref:
                # 触发异步 reset，不等待完成
                env_ref.reset.remote()
            else:
                logger.warning(f"Could not get env_ref for {env_id}")

        logger.info(f"Successfully created and triggered reset for {request.count} environments")
        return BatchCreateResponse(env_ids=env_ids)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create environments: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create environments: {str(e)}"
        )


@app.post("/envs/{env_id}/reset", response_model=ResetResponse)
async def reset_env(env_id: str):
    """Reset环境"""
    logger.info(f"Resetting environment {env_id[:8]}...")

    try:
        env_pool = get_global_env_pool()
        env_ref = ray.get(env_pool.get_env.remote(env_id))

        if env_ref is None:
            logger.warning(f"Environment {env_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        reset_start = time.time()
        result = ray.get(env_ref.reset.remote())
        reset_duration = time.time() - reset_start

        # MCEnvActor.reset() 只返回 obs，不是 (obs, info)
        # 兼容处理
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        logger.info(f"Environment {env_id[:8]}... reset completed in {reset_duration:.2f}s")

        return ResetResponse(
            observation=serialize_observation(obs),
            info=serialize_value(info)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset env {env_id[:8]}...: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset: {str(e)}"
        )


@app.get("/envs/{env_id}/reset_result", response_model=ResetResponse)
async def get_reset_result(env_id: str, wait: int = 0):
    """获取后台 reset 的结果（用于 batch_create_envs 后的异步 reset）

    Args:
        env_id: 环境ID
        wait: 等待时间（秒），如果reset未完成，最多等待这么多秒
    """
    logger.debug(f"Getting reset result for env {env_id[:8]}... (wait={wait}s)")

    try:
        env_pool = get_global_env_pool()
        env_ref = ray.get(env_pool.get_env.remote(env_id))

        if env_ref is None:
            logger.warning(f"Environment {env_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        # 轮询等待 reset 完成
        start_time = time.time()
        poll_count = 0
        while True:
            obs, info = ray.get(env_ref.get_reset_result.remote())

            if obs is not None:
                # reset 完成
                elapsed = time.time() - start_time
                logger.info(
                    f"Env {env_id[:8]}... reset result ready after {elapsed:.2f}s "
                    f"({poll_count} polls)"
                )
                return ResetResponse(
                    observation=serialize_observation(obs),
                    info=serialize_value(info)
                )

            # 检查是否超时
            elapsed = time.time() - start_time
            if elapsed >= wait:
                # 返回202表示reset正在进行中
                logger.debug(
                    f"Env {env_id[:8]}... reset still in progress after {elapsed:.2f}s "
                    f"({poll_count} polls)"
                )
                raise HTTPException(
                    status_code=status.HTTP_202_ACCEPTED,
                    detail=f"Environment {env_id} reset is still in progress"
                )

            # 等待一小段时间后重试
            poll_count += 1
            await asyncio.sleep(1)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get reset result for env {env_id[:8]}...: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get reset result: {str(e)}"
        )


@app.post("/envs/{env_id}/step", response_model=StepResponse)
async def step_env(env_id: str, request: StepRequest):
    """Step环境"""
    try:
        env_pool = get_global_env_pool()
        env_ref = ray.get(env_pool.get_env.remote(env_id))

        if env_ref is None:
            logger.warning(f"Environment {env_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        # 记录action类型（不记录完整内容，避免日志过大）
        action_type = type(request.action).__name__
        logger.debug(f"Stepping env {env_id[:8]}... with action type={action_type}")

        step_start = time.time()
        result = ray.get(env_ref.step.remote(request.action))
        step_duration = time.time() - step_start

        # MCEnvActor.step() 返回旧版Gym格式 (obs, reward, done, info)
        # 需要转换为新版格式 (obs, reward, terminated, truncated, info)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False  # 旧版没有truncated，默认False
        else:
            obs, reward, terminated, truncated, info = result

        # 只在结束或错误时记录详细信息
        if terminated or truncated:
            logger.info(
                f"Env {env_id[:8]}... episode ended - "
                f"terminated={terminated}, truncated={truncated}, "
                f"reward={reward:.2f}, duration={step_duration:.3f}s"
            )

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
        logger.error(f"Failed to step env {env_id[:8]}...: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to step: {str(e)}"
        )


@app.delete("/envs/{env_id}", status_code=status.HTTP_204_NO_CONTENT)
async def close_env(env_id: str):
    """关闭环境"""
    logger.info(f"Closing environment {env_id[:8]}...")

    try:
        env_pool = get_global_env_pool()
        success = ray.get(env_pool.close_env.remote(env_id))

        if not success:
            logger.warning(f"Environment {env_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {env_id} not found"
            )

        logger.info(f"Environment {env_id[:8]}... closed successfully")
        return None

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to close env {env_id[:8]}...: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to close: {str(e)}"
        )


# ============================================================================
# 全局异常处理
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """捕获所有未处理的异常"""
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {str(exc)}",
        exc_info=True
    )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Internal server error: {str(exc)}"
    )


# ============================================================================
# 启动
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting uvicorn server...")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=False  # 使用我们自己的请求日志中间件
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Server crashed: {str(e)}", exc_info=True)
        raise
