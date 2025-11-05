import pytest
import requests

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
