"""
测试双action格式支持
支持LLM格式（字符串）和Agent格式（dict）
"""

import pytest
import requests
import numpy as np
from raycraft.http_client import RemoteEnv
from concurrent.futures import ThreadPoolExecutor

SERVER_URL = "http://localhost:8000"

# ============================================================================
# 测试辅助函数
# ============================================================================

def create_env_ids(server_url: str, count: int, env_kwargs=None):
    """创建环境并返回ID列表"""
    if env_kwargs is None:
        env_kwargs = {}

    resp = requests.post(
        f"{server_url}/batch/envs",
        json={"count": count, "env_name": "minecraft", "env_kwargs": env_kwargs},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["env_ids"]

def assert_step_result(obs, reward, terminated, truncated, info):
    """验证step返回值格式"""
    assert isinstance(obs, dict), f"obs应该是dict，实际是{type(obs)}"
    assert isinstance(reward, (int, float)), f"reward应该是数字，实际是{type(reward)}"
    assert isinstance(terminated, bool), f"terminated应该是bool，实际是{type(terminated)}"
    assert isinstance(truncated, bool), f"truncated应该是bool，实际是{type(truncated)}"
    assert isinstance(info, dict), f"info应该是dict，实际是{type(info)}"

# ============================================================================
# 测试 1: 基础LLM格式兼容性
# ============================================================================

def test_llm_format_basic():
    """测试LLM格式基础功能"""
    env_id = create_env_ids(SERVER_URL, count=1)[0]

    with RemoteEnv(SERVER_URL, env_id) as env:
        # Reset
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

        # Step
        obs, reward, term, trunc, info = env.step('[{"action": "forward"}]')
        assert_step_result(obs, reward, term, trunc, info)

# ============================================================================
# 测试 2: Agent格式基础功能
# ============================================================================

def test_agent_format_basic():
    """测试Agent格式基础功能"""
    env_id = create_env_ids(SERVER_URL, count=1)[0]

    with RemoteEnv(SERVER_URL, env_id) as env:
        # Reset
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

        # Step - 使用有效的索引值
        action = {'buttons': 5, 'camera': 60}
        obs, reward, term, trunc, info = env.step(action)
        assert_step_result(obs, reward, term, trunc, info)

def test_agent_format_various_actions():
    """测试Agent格式不同动作"""
    env_id = create_env_ids(SERVER_URL, count=1)[0]

    with RemoteEnv(SERVER_URL, env_id) as env:
        obs, info = env.reset()

        # 测试不同的buttons和camera组合
        test_actions = [
            {'buttons': 0, 'camera': 60},     # 无动作，中心视角
            {'buttons': 1, 'camera': 45},     # 左转
            {'buttons': 2, 'camera': 80},     # 右转
            {'buttons': 3, 'camera': 100},    # 前进
            {'buttons': 4, 'camera': 50},     # 后退
            {'buttons': 5, 'camera': 75},     # 跳跃
            {'buttons': 6, 'camera': 90},     # 攻击
        ]

        for action in test_actions:
            obs, reward, term, trunc, info = env.step(action)
            assert_step_result(obs, reward, term, trunc, info)

# ============================================================================
# 测试 3: 混合格式测试
# ============================================================================

def test_mixed_format_in_episode():
    """在同一个episode中混合使用两种格式"""
    env_id = create_env_ids(SERVER_URL, count=1)[0]

    with RemoteEnv(SERVER_URL, env_id) as env:
        obs, info = env.reset()

        # 交替使用两种格式
        actions = [
            '[{"action": "forward"}]',              # LLM格式
            {'buttons': 5, 'camera': 60},           # Agent格式
            '[{"action": "jump"}]',                 # LLM格式
            {'buttons': 1, 'camera': 45},           # Agent格式
            '[{"action": "left"}]',                 # LLM格式
            {'buttons': 3, 'camera': 80},           # Agent格式
        ]

        for action in actions:
            obs, reward, term, trunc, info = env.step(action)
            assert_step_result(obs, reward, term, trunc, info)

def test_mixed_format_parallel():
    """并行环境使用不同格式"""
    env_ids = create_env_ids(SERVER_URL, count=4)

    def run_llm_format(env_id):
        """运行LLM格式环境"""
        with RemoteEnv(SERVER_URL, env_id) as env:
            obs, info = env.reset()
            for _ in range(5):
                obs, reward, term, trunc, info = env.step('[{"action": "forward"}]')

    def run_agent_format(env_id):
        """运行Agent格式环境"""
        with RemoteEnv(SERVER_URL, env_id) as env:
            obs, info = env.reset()
            for _ in range(5):
                obs, reward, term, trunc, info = env.step({'buttons': 3, 'camera': 60})

    # 并行运行
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_llm_format, env_ids[0]),
            executor.submit(run_agent_format, env_ids[1]),
            executor.submit(run_llm_format, env_ids[2]),
            executor.submit(run_agent_format, env_ids[3]),
        ]

        # 等待所有完成
        for future in futures:
            future.result()  # 会抛出异常如果有失败

# ============================================================================
# 测试 4: 边界条件和错误处理
# ============================================================================

def test_invalid_action_types():
    """测试无效的action类型"""
    env_id = create_env_ids(SERVER_URL, count=1)[0]

    with RemoteEnv(SERVER_URL, env_id) as env:
        obs, info = env.reset()

        # 测试各种无效的action格式
        invalid_actions = [
            None,                                   # None
            123,                                   # 数字
            [],                                    # 空列表
            "invalid",                             # 无效字符串
            {'wrong': 'format'},                   # 字典格式错误
            {'buttons': 9999, 'camera': 200},     # 索引超出范围
        ]

        for action in invalid_actions:
            with pytest.raises(Exception):  # 应该抛出某种异常
                env.step(action)

def test_agent_format_index_validation():
    """测试Agent格式索引验证"""
    env_id = create_env_ids(SERVER_URL, count=1)[0]

    with RemoteEnv(SERVER_URL, env_id) as env:
        obs, info = env.reset()

        # 测试边界索引
        valid_actions = [
            {'buttons': 0, 'camera': 0},           # 最小值
            {'buttons': 1023, 'camera': 120},       # 最大值
            {'buttons': 512, 'camera': 60},        # 中间值
        ]

        for action in valid_actions:
            obs, reward, term, trunc, info = env.step(action)
            assert_step_result(obs, reward, term, trunc, info)

        # 测试超出范围的索引
        invalid_actions = [
            {'buttons': -1, 'camera': 60},         # 负数
            {'buttons': 1024, 'camera': 60},        # buttons超出范围
            {'buttons': 100, 'camera': -1},        # camera负数
            {'buttons': 100, 'camera': 121},       # camera超出范围
        ]

        for action in invalid_actions:
            with pytest.raises(Exception):
                env.step(action)

def test_agent_format_nested_types():
    """测试Agent格式的嵌套类型"""
    env_id = create_env_ids(SERVER_URL, count=1)[0]

    with RemoteEnv(SERVER_URL, env_id) as env:
        obs, info = env.reset()

        # 测试不同数值类型（应该被正确转换）
        valid_actions = [
            {'buttons': 5, 'camera': 60},           # int
            {'buttons': 5.0, 'camera': 60.0},       # float
            {'buttons': np.int32(5), 'camera': np.int32(60)},  # numpy
        ]

        for action in valid_actions:
            obs, reward, term, trunc, info = env.step(action)
            assert_step_result(obs, reward, term, trunc, info)

# ============================================================================
# 测试 5: 性能测试
# ============================================================================

def test_format_switching_performance():
    """测试格式切换的性能"""
    import time

    env_id = create_env_ids(SERVER_URL, count=1)[0]

    with RemoteEnv(SERVER_URL, env_id) as env:
        obs, info = env.reset()

        # 测试LLM格式性能
        start_time = time.time()
        for _ in range(20):
            env.step('[{"action": "forward"}]')
        llm_time = time.time() - start_time

        # 测试Agent格式性能
        start_time = time.time()
        for _ in range(20):
            env.step({'buttons': 3, 'camera': 60})
        agent_time = time.time() - start_time

        # 性能差异不应该太大（允许2倍差异）
        assert max(llm_time, agent_time) / min(llm_time, agent_time) < 3.0, \
            f"性能差异过大: LLM={llm_time:.3f}s, Agent={agent_time:.3f}s"

# ============================================================================
# 测试 6: 向后兼容性
# ============================================================================

def test_backward_compatibility():
    """确保向后兼容性"""
    env_id = create_env_ids(SERVER_URL, count=1)[0]

    with RemoteEnv(SERVER_URL, env_id) as env:
        obs, info = env.reset()

        # 测试所有原有的LLM格式action
        llm_actions = [
            '[{"action": "forward"}]',
            '[{"action": "back"}]',
            '[{"action": "left"}]',
            '[{"action": "right"}]',
            '[{"action": "jump"}]',
            '[{"action": "attack"}]',
            '[{"action": "use"}]',
            '[{"action": "sneak"}]',
            '[{"action": "sprint"}]',
        ]

        for action in llm_actions:
            obs, reward, term, trunc, info = env.step(action)
            assert_step_result(obs, reward, term, trunc, info)

# ============================================================================
# 测试 7: 集成测试
# ============================================================================

def test_full_workflow_with_yaml_config():
    """使用YAML配置的完整工作流测试"""
    # 使用启用录制的YAML配置
    yaml_config = "configs/simple_record.yaml"
    env_id = create_env_ids(SERVER_URL, count=1, env_kwargs=yaml_config)[0]

    with RemoteEnv(SERVER_URL, env_id) as env:
        obs, info = env.reset()

        # 混合使用两种格式
        actions = [
            '[{"action": "forward"}]',              # LLM格式
            {'buttons': 5, 'camera': 60},           # Agent格式
            '[{"action": "jump"}]',                 # LLM格式
            {'buttons': 1, 'camera': 45},           # Agent格式
        ]

        for action in actions:
            obs, reward, term, trunc, info = env.step(action)
            assert_step_result(obs, reward, term, trunc, info)

# ============================================================================
# 测试运行器
# ============================================================================

if __name__ == "__main__":
    """运行所有测试"""
    # 检查服务器
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code != 200:
            print(f"✗ 服务器不可用: {SERVER_URL}")
            print("  请先启动服务器: python -m raycraft.http_server")
            exit(1)
        print(f"✓ 服务器已连接: {SERVER_URL}")
    except Exception as e:
        print(f"✗ 无法连接到服务器: {e}")
        exit(1)

    # 运行测试
    import pytest
    import sys

    # 添加当前项目路径
    sys.path.insert(0, '/fs-computility-new/nuclear/leishanzhe/repo/raycraft')

    # 运行这个文件中的所有测试
    pytest.main([__file__, "-v", "--tb=short"])