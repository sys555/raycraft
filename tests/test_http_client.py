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
