"""
RayCraft HTTP Client 使用示例

前提条件：
1. 在服务器机器上启动 HTTP Server：
   python -m raycraft.http_server

2. 在客户端机器上只需要安装 requests：
   pip install requests
"""

import requests
from raycraft.http_client import RemoteEnv
from concurrent.futures import ThreadPoolExecutor

# 服务器地址（替换为实际地址）
SERVER_URL = "http://localhost:8000"
# SERVER_URL = "http://10.0.1.100:8000"  # 如果服务器在远程


# ============================================================================
# 辅助函数：批量创建环境ID
# ============================================================================

def create_env_ids(server_url: str, count: int):
    """调用批量创建API获取环境ID列表"""
    resp = requests.post(
        f"{server_url}/batch/envs",
        json={"count": count, "env_name": "minecraft", "env_kwargs": {}},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()["env_ids"]


# ============================================================================
# 示例 1: 基础使用
# ============================================================================

def example_basic():
    """最简单的使用方式"""
    print("\n=== 示例 1: 基础使用 ===")

    # 1. 获取环境ID
    env_ids = create_env_ids(SERVER_URL, count=1)
    env_id = env_ids[0]

    # 2. 创建RemoteEnv对象
    env = RemoteEnv(SERVER_URL, env_id)

    # Reset
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"✓ Reset完成，observation keys: {list(obs.keys())}")

    # Step 10步
    print("Running 10 steps...")
    for i in range(10):
        obs, reward, terminated, truncated, info = env.step('[{"action": "forward"}]')
        print(f"  Step {i+1}: reward={reward:.2f}, terminated={terminated}")

        if terminated or truncated:
            print("  Episode结束，重新reset")
            obs, info = env.reset()

    # Close
    env.close()
    print("✓ 环境已关闭\n")


# ============================================================================
# 示例 2: 上下文管理器（推荐）
# ============================================================================

def example_context_manager():
    """使用 with 语句自动管理环境生命周期"""
    print("\n=== 示例 2: 上下文管理器 ===")

    # 获取环境ID并创建RemoteEnv
    env_id = create_env_ids(SERVER_URL, count=1)[0]

    # with 自动调用 close()
    with RemoteEnv(SERVER_URL, env_id) as env:
        obs, info = env.reset()
        print(f"✓ Reset完成")

        for i in range(5):
            obs, reward, term, trunc, info = env.step('[{"action": "forward"}]')
            print(f"  Step {i+1}: reward={reward:.2f}")

    print("✓ with 退出后环境自动关闭\n")


# ============================================================================
# 示例 3: 批量并行环境
# ============================================================================

def example_batch_parallel():
    """批量创建环境并并行运行"""
    print("\n=== 示例 3: 批量并行环境 ===")

    # 批量创建4个环境
    num_envs = 4
    env_ids = create_env_ids(SERVER_URL, count=num_envs)
    envs = [RemoteEnv(SERVER_URL, env_id) for env_id in env_ids]
    print(f"✓ 创建了 {num_envs} 个环境")

    def run_episode(env_idx, env):
        """单个环境的运行逻辑"""
        obs, info = env.reset()
        total_reward = 0

        for step in range(20):
            obs, reward, term, trunc, info = env.step('[{"action": "forward"}]')
            total_reward += reward

            if term or trunc:
                break

        env.close()
        return env_idx, total_reward

    # 使用线程池并行运行
    with ThreadPoolExecutor(max_workers=num_envs) as executor:
        futures = [
            executor.submit(run_episode, i, env)
            for i, env in enumerate(envs)
        ]

        results = [f.result() for f in futures]

    print("\n结果:")
    for env_idx, total_reward in results:
        print(f"  环境 {env_idx}: 总奖励 = {total_reward:.2f}")
    print()


# ============================================================================
# 示例 4: 自定义 action
# ============================================================================

def example_custom_actions():
    """使用不同的 action"""
    print("\n=== 示例 4: 自定义 action ===")

    env_id = create_env_ids(SERVER_URL, count=1)[0]

    with RemoteEnv(SERVER_URL, env_id) as env:
        obs, info = env.reset()
        print("✓ Reset完成")

        # 不同的 action 示例
        actions = [
            '[{"action": "forward"}]',      # 前进
            '[{"action": "back"}]',         # 后退
            '[{"action": "left"}]',         # 左转
            '[{"action": "right"}]',        # 右转
            '[{"action": "jump"}]',         # 跳跃
            '[{"action": "attack"}]',       # 攻击
        ]

        for action in actions:
            obs, reward, term, trunc, info = env.step(action)
            print(f"  Action: {action:30s} -> reward={reward:.2f}")

    print()


# ============================================================================
# 示例 5: 错误处理
# ============================================================================

def example_error_handling():
    """演示错误处理"""
    print("\n=== 示例 5: 错误处理 ===")

    try:
        # 尝试连接不存在的服务器
        env_ids = create_env_ids("http://invalid-server:8000", count=1)
    except requests.exceptions.ConnectionError as e:
        print(f"✗ 连接错误: 无法连接到服务器")
    except requests.exceptions.Timeout as e:
        print(f"✗ 超时错误: 服务器响应超时")
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP错误: {e.response.status_code}")
    except Exception as e:
        print(f"✗ 未知错误: {type(e).__name__}: {e}")

    print()


# ============================================================================
# 示例 6: 处理 RGB 图像
# ============================================================================

def example_rgb_images():
    """处理返回的 RGB 图像"""
    print("\n=== 示例 6: 处理 RGB 图像 ===")

    env_id = create_env_ids(SERVER_URL, count=1)[0]

    with RemoteEnv(SERVER_URL, env_id) as env:
        obs, info = env.reset()

        # observation['rgb'] 是压缩后的格式
        if 'rgb' in obs:
            rgb_data = obs['rgb']
            print(f"✓ RGB 数据:")
            print(f"  类型: {rgb_data.get('type', 'unknown')}")
            print(f"  数据大小: {len(rgb_data.get('data', ''))} bytes (base64)")

            # 如果需要解码图像：
            # import base64
            # import io
            # from PIL import Image
            #
            # jpeg_bytes = base64.b64decode(rgb_data['data'])
            # image = Image.open(io.BytesIO(jpeg_bytes))
            # image.save('screenshot.jpg')

        print()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("=" * 70)
    print("RayCraft HTTP Client 使用示例")
    print("=" * 70)

    try:
        # 检查服务器是否可用
        import requests
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code != 200:
            print(f"\n✗ 服务器不可用: {SERVER_URL}")
            print(f"  请先启动服务器: python -m raycraft.http_server\n")
            return

        print(f"\n✓ 服务器已连接: {SERVER_URL}")
        health = resp.json()
        print(f"  Ray 状态: {health['ray_initialized']}")
        print(f"  当前环境数: {health['num_environments']}")

        # 运行示例（注释掉不需要的）
        example_basic()
        example_context_manager()
        # example_batch_parallel()  # 这个比较慢，需要并行reset多个环境
        example_custom_actions()
        example_error_handling()
        example_rgb_images()

        print("=" * 70)
        print("所有示例运行完成！")
        print("=" * 70)

    except requests.exceptions.ConnectionError:
        print(f"\n✗ 无法连接到服务器: {SERVER_URL}")
        print(f"  请检查：")
        print(f"  1. 服务器是否已启动: python -m raycraft.http_server")
        print(f"  2. 服务器地址是否正确")
        print(f"  3. 网络是否可达\n")
    except KeyboardInterrupt:
        print("\n\n中断运行\n")
    except Exception as e:
        print(f"\n✗ 错误: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
