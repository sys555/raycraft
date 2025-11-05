# RayCraft HTTP API 实现文档（TDD）

## TDD 开发流程

```
1. RED   → 写测试（必然失败）
2. GREEN → 写最少代码让测试通过
3. REFACTOR → 重构代码
```

按顺序实现：
1. EnvPool 扩展方法
2. HTTP Server
3. HTTP Client

---

## 第一步：EnvPool 扩展（已完成）

EnvPool 已经添加了 HTTP API 需要的方法，见 `raycraft/ray/pool.py:329-388`

跳过这一步。

---

## 第二步：HTTP Server

### 架构

```
FastAPI Server
    ↓
GlobalEnvPool (Ray Named Actor)
    ↓
MCEnvActor × N
```

### 测试先行（tests/test_http_server.py）

```python
import pytest
import requests
from fastapi.testclient import TestClient

# 如果需要真实Ray环境，用requests
# 如果只测试FastAPI逻辑，用TestClient

SERVER = "http://localhost:8000"

# ============================================================================
# 测试 1: 健康检查
# ============================================================================

def test_health_check():
    """RED: 服务还没实现，测试必然失败"""
    resp = requests.get(f"{SERVER}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["ray_initialized"] == True


# ============================================================================
# 测试 2: 批量创建环境
# ============================================================================

def test_batch_create_envs():
    """RED: 批量创建接口还没实现"""
    resp = requests.post(
        f"{SERVER}/batch/envs",
        json={"count": 3, "env_name": "minecraft", "env_kwargs": {}}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "env_ids" in data
    assert len(data["env_ids"]) == 3
    # 清理
    for env_id in data["env_ids"]:
        requests.delete(f"{SERVER}/envs/{env_id}")


def test_batch_create_zero_envs():
    """边界情况：创建0个环境"""
    resp = requests.post(
        f"{SERVER}/batch/envs",
        json={"count": 0, "env_name": "minecraft", "env_kwargs": {}}
    )
    assert resp.status_code == 400  # Bad Request


def test_batch_create_invalid_count():
    """边界情况：负数count"""
    resp = requests.post(
        f"{SERVER}/batch/envs",
        json={"count": -1, "env_name": "minecraft", "env_kwargs": {}}
    )
    assert resp.status_code == 400


# ============================================================================
# 测试 3: Reset 环境
# ============================================================================

def test_reset_env():
    """RED: reset接口还没实现"""
    # 先创建环境
    resp = requests.post(
        f"{SERVER}/batch/envs",
        json={"count": 1, "env_name": "minecraft", "env_kwargs": {}}
    )
    env_id = resp.json()["env_ids"][0]

    # Reset
    resp = requests.post(f"{SERVER}/envs/{env_id}/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert "observation" in data
    assert "info" in data

    # 清理
    requests.delete(f"{SERVER}/envs/{env_id}")


def test_reset_nonexistent_env():
    """错误情况：reset不存在的环境"""
    resp = requests.post(f"{SERVER}/envs/fake-uuid-123/reset")
    assert resp.status_code == 404


# ============================================================================
# 测试 4: Step 环境
# ============================================================================

def test_step_env():
    """RED: step接口还没实现"""
    # 创建并reset
    resp = requests.post(
        f"{SERVER}/batch/envs",
        json={"count": 1, "env_name": "minecraft", "env_kwargs": {}}
    )
    env_id = resp.json()["env_ids"][0]
    requests.post(f"{SERVER}/envs/{env_id}/reset")

    # Step
    resp = requests.post(
        f"{SERVER}/envs/{env_id}/step",
        json={"action": '[{"action": "forward"}]'}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "observation" in data
    assert "reward" in data
    assert "terminated" in data
    assert "truncated" in data
    assert "info" in data
    assert isinstance(data["reward"], (int, float))
    assert isinstance(data["terminated"], bool)
    assert isinstance(data["truncated"], bool)

    # 清理
    requests.delete(f"{SERVER}/envs/{env_id}")


def test_step_nonexistent_env():
    """错误情况：step不存在的环境"""
    resp = requests.post(
        f"{SERVER}/envs/fake-uuid-123/step",
        json={"action": '[{"action": "forward"}]'}
    )
    assert resp.status_code == 404


def test_step_invalid_action():
    """错误情况：无效的action格式"""
    resp = requests.post(
        f"{SERVER}/batch/envs",
        json={"count": 1, "env_name": "minecraft", "env_kwargs": {}}
    )
    env_id = resp.json()["env_ids"][0]
    requests.post(f"{SERVER}/envs/{env_id}/reset")

    # 错误的action格式
    resp = requests.post(
        f"{SERVER}/envs/{env_id}/step",
        json={"action": "not-a-json-string"}
    )
    # 应该返回400或500，但不会404
    assert resp.status_code in [400, 500]

    requests.delete(f"{SERVER}/envs/{env_id}")


# ============================================================================
# 测试 5: Close 环境
# ============================================================================

def test_close_env():
    """RED: close接口还没实现"""
    # 创建环境
    resp = requests.post(
        f"{SERVER}/batch/envs",
        json={"count": 1, "env_name": "minecraft", "env_kwargs": {}}
    )
    env_id = resp.json()["env_ids"][0]

    # Close
    resp = requests.delete(f"{SERVER}/envs/{env_id}")
    assert resp.status_code == 204  # No Content

    # 再次close应该404
    resp = requests.delete(f"{SERVER}/envs/{env_id}")
    assert resp.status_code == 404


def test_close_nonexistent_env():
    """错误情况：close不存在的环境"""
    resp = requests.delete(f"{SERVER}/envs/fake-uuid-123")
    assert resp.status_code == 404


# ============================================================================
# 测试 6: 完整工作流
# ============================================================================

def test_full_workflow():
    """完整的create→reset→step→close流程"""
    # 批量创建
    resp = requests.post(
        f"{SERVER}/batch/envs",
        json={"count": 2, "env_name": "minecraft", "env_kwargs": {}}
    )
    assert resp.status_code == 200
    env_ids = resp.json()["env_ids"]
    assert len(env_ids) == 2

    for env_id in env_ids:
        # Reset
        resp = requests.post(f"{SERVER}/envs/{env_id}/reset")
        assert resp.status_code == 200

        # Step 10次
        for _ in range(10):
            resp = requests.post(
                f"{SERVER}/envs/{env_id}/step",
                json={"action": '[{"action": "forward"}]'}
            )
            assert resp.status_code == 200

        # Close
        resp = requests.delete(f"{SERVER}/envs/{env_id}")
        assert resp.status_code == 204


# ============================================================================
# 测试 7: 并发安全
# ============================================================================

def test_concurrent_requests():
    """多个请求并发访问同一个环境"""
    from concurrent.futures import ThreadPoolExecutor

    # 创建1个环境
    resp = requests.post(
        f"{SERVER}/batch/envs",
        json={"count": 1, "env_name": "minecraft", "env_kwargs": {}}
    )
    env_id = resp.json()["env_ids"][0]
    requests.post(f"{SERVER}/envs/{env_id}/reset")

    # 并发step
    def do_step(i):
        resp = requests.post(
            f"{SERVER}/envs/{env_id}/step",
            json={"action": '[{"action": "forward"}]'}
        )
        return resp.status_code == 200

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(do_step, range(20)))

    assert all(results)  # 所有请求都应该成功

    requests.delete(f"{SERVER}/envs/{env_id}")
```

### GREEN: 实现代码（raycraft/http_server.py）

```python
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
    env_ids: list[str]


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
            result[key] = value
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

        obs, info = ray.get(env_ref.reset.remote())

        return ResetResponse(
            observation=serialize_observation(obs),
            info=info
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

        obs, reward, terminated, truncated, info = ray.get(
            env_ref.step.remote(request.action)
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
```

### REFACTOR: 优化点

1. 添加日志
2. 添加 CORS 支持（如果需要跨域）
3. 添加请求限流
4. 添加 Prometheus metrics

先让测试通过，再考虑这些。

---

## 第三步：HTTP Client

### 测试先行（tests/test_http_client.py）

```python
import pytest
from raycraft.http_client import RemoteEnv, create_remote_envs

SERVER = "http://localhost:8000"  # 需要先启动服务器

# ============================================================================
# 测试 1: 批量创建
# ============================================================================

def test_create_remote_envs():
    """RED: create_remote_envs还没实现"""
    envs = create_remote_envs(SERVER, count=3)
    assert len(envs) == 3
    assert all(isinstance(env, RemoteEnv) for env in envs)

    # 清理
    for env in envs:
        env.close()


def test_create_remote_envs_with_kwargs():
    """带参数创建"""
    envs = create_remote_envs(SERVER, count=1, env_kwargs={"seed": 42})
    assert len(envs) == 1
    envs[0].close()


# ============================================================================
# 测试 2: RemoteEnv.reset
# ============================================================================

def test_remote_env_reset():
    """RED: RemoteEnv.reset还没实现"""
    envs = create_remote_envs(SERVER, count=1)
    env = envs[0]

    obs, info = env.reset()
    assert isinstance(obs, dict)
    assert isinstance(info, dict)

    env.close()


# ============================================================================
# 测试 3: RemoteEnv.step
# ============================================================================

def test_remote_env_step():
    """RED: RemoteEnv.step还没实现"""
    envs = create_remote_envs(SERVER, count=1)
    env = envs[0]

    env.reset()
    obs, reward, terminated, truncated, info = env.step('[{"action": "forward"}]')

    assert isinstance(obs, dict)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env.close()


# ============================================================================
# 测试 4: RemoteEnv.close
# ============================================================================

def test_remote_env_close():
    """RED: RemoteEnv.close还没实现"""
    envs = create_remote_envs(SERVER, count=1)
    env = envs[0]

    # 第一次close应该成功
    env.close()

    # 第二次close应该失败（环境已经不存在）
    with pytest.raises(Exception):
        env.close()


# ============================================================================
# 测试 5: 完整工作流
# ============================================================================

def test_remote_env_full_workflow():
    """完整流程"""
    envs = create_remote_envs(SERVER, count=2)

    for env in envs:
        # Reset
        obs, info = env.reset()
        assert obs is not None

        # Step 10次
        for _ in range(10):
            obs, reward, done, truncated, info = env.step('[{"action": "forward"}]')

        # Close
        env.close()


# ============================================================================
# 测试 6: 上下文管理器
# ============================================================================

def test_remote_env_context_manager():
    """RED: 上下文管理器还没实现"""
    envs = create_remote_envs(SERVER, count=1)

    with envs[0] as env:
        obs, info = env.reset()
        obs, reward, done, truncated, info = env.step('[{"action": "forward"}]')

    # 退出with后应该自动close


# ============================================================================
# 测试 7: 并行使用
# ============================================================================

def test_parallel_remote_envs():
    """多个RemoteEnv并行使用"""
    from concurrent.futures import ThreadPoolExecutor

    envs = create_remote_envs(SERVER, count=8)

    def run_env(env):
        env.reset()
        for _ in range(10):
            env.step('[{"action": "forward"}]')
        env.close()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run_env, envs))

    # 所有环境都应该正常运行
```

### GREEN: 实现代码（raycraft/http_client.py）

```python
"""
RayCraft HTTP Client

只依赖：
- requests (HTTP库)
- 标准库 (typing, json等)

不依赖：
- ray
- numpy
- PIL
- minestudio
"""

import requests
from typing import Tuple, Dict, Any, List, Optional


class RemoteEnv:
    """远程环境客户端（轻量级）"""

    def __init__(self, server_url: str, env_id: str):
        """
        Args:
            server_url: 服务器地址，如 "http://10.0.1.100:8000"
            env_id: 环境UUID
        """
        self.server_url = server_url.rstrip('/')
        self.env_id = env_id
        self.session = requests.Session()  # 连接复用

    def reset(self, timeout: int = 120) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset环境

        Args:
            timeout: 超时时间（秒），首次reset慢，建议120秒

        Returns:
            (observation, info)
        """
        resp = self.session.post(
            f"{self.server_url}/envs/{self.env_id}/reset",
            timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return data["observation"], data["info"]

    def step(
        self,
        action: str,
        timeout: int = 10
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Step环境

        Args:
            action: JSON格式的action字符串，如 '[{"action": "forward"}]'
            timeout: 超时时间（秒）

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        resp = self.session.post(
            f"{self.server_url}/envs/{self.env_id}/step",
            json={"action": action},
            timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data["observation"],
            data["reward"],
            data["terminated"],
            data["truncated"],
            data["info"]
        )

    def close(self, timeout: int = 30):
        """关闭环境

        Args:
            timeout: 超时时间（秒），需要等待MP4保存
        """
        resp = self.session.delete(
            f"{self.server_url}/envs/{self.env_id}",
            timeout=timeout
        )
        resp.raise_for_status()
        self.session.close()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出，自动close"""
        try:
            self.close()
        except:
            pass  # 忽略close错误


def create_remote_envs(
    server_url: str,
    count: int,
    env_kwargs: Optional[Dict[str, Any]] = None
) -> List[RemoteEnv]:
    """批量创建RemoteEnv对象

    Args:
        server_url: 服务器地址
        count: 环境数量
        env_kwargs: 环境参数

    Returns:
        RemoteEnv对象列表
    """
    resp = requests.post(
        f"{server_url.rstrip('/')}/batch/envs",
        json={
            "count": count,
            "env_name": "minecraft",
            "env_kwargs": env_kwargs or {}
        },
        timeout=60
    )
    resp.raise_for_status()
    env_ids = resp.json()["env_ids"]

    return [RemoteEnv(server_url, env_id) for env_id in env_ids]
```

### REFACTOR: 优化点

1. 添加重试机制（网络抖动）
2. 添加连接池配置
3. 添加错误处理和日志

先让测试通过。

---

## 运行测试

### 1. 启动服务器

```bash
# 终端1
python -m raycraft.http_server
```

### 2. 运行服务端测试

```bash
# 终端2
pytest tests/test_http_server.py -v
```

### 3. 运行客户端测试

```bash
pytest tests/test_http_client.py -v
```

### 4. 运行所有测试

```bash
pytest tests/test_http*.py -v
```

---

## TDD 检查清单

- [ ] 所有测试都写在实现之前
- [ ] 测试覆盖正常流程
- [ ] 测试覆盖错误情况
- [ ] 测试覆盖边界情况
- [ ] 每个测试独立（可以单独运行）
- [ ] 测试可重复（运行多次结果一致）
- [ ] 测试有清理逻辑（不留垃圾数据）
- [ ] 代码只实现让测试通过的最少功能
- [ ] 通过测试后进行重构
- [ ] 重构后测试仍然通过

完了。
