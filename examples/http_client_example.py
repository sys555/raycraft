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

def create_env_ids(server_url: str, count: int, env_kwargs=None):
    """调用批量创建API获取环境ID列表

    Args:
        server_url: 服务器地址
        count: 环境数量
        env_kwargs: 环境配置，可以是:
            - dict: 配置字典
            - str: YAML配置文件路径
            - None: 使用默认配置
    """
    if env_kwargs is None:
        env_kwargs = {}

    resp = requests.post(
        f"{server_url}/batch/envs",
        json={"count": count, "env_name": "minecraft", "env_kwargs": env_kwargs},
        timeout=120  # 增加超时：批量创建会等待所有环境reset完成
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
    """批量创建环境并并行运行（使用YAML配置）"""
    print("\n=== 示例 3: 批量并行环境（带视频录制）===")

    # 使用YAML配置批量创建2个环境（并行启动Minecraft进程资源密集）
    num_envs = 2
    yaml_config = "configs/kill/kill_zombie_with_record.yaml"
    print(f"使用配置: {yaml_config}")

    env_ids = create_env_ids(SERVER_URL, count=num_envs, env_kwargs=yaml_config)
    envs = [RemoteEnv(SERVER_URL, env_id) for env_id in env_ids]
    print(f"✓ 创建了 {num_envs} 个环境")

    def run_episode(env_idx, env):
        """单个环境的运行逻辑 - 测试双action格式"""
        # 注意：batch_create_envs 已在后台自动执行 reset，无需首次 reset
        total_reward = 0
        step_count = 0

        # 交替使用LLM格式和Agent格式
        for step in range(10):
            # 偶数步用LLM格式，奇数步用Agent格式
            if step % 2 == 0:
                action = '[{"action": "forward"}]'
            else:
                action = {'buttons': 3, 'camera': 60}

            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            step_count += 1

            # if term or trunc:
            #     break

        # 触发视频保存
        env.reset(timeout=180)
        env.close()
        return env_idx, total_reward, step_count

    # 使用线程池并行运行
    print("\n并行运行环境...")
    with ThreadPoolExecutor(max_workers=num_envs) as executor:
        futures = [
            executor.submit(run_episode, i, env)
            for i, env in enumerate(envs)
        ]

        results = [f.result() for f in futures]

    print("\n结果:")
    for env_idx, total_reward, steps in results:
        print(f"  环境 {env_idx}: 总奖励 = {total_reward:.2f}, 步数 = {steps}")
    print(f"✓ 所有视频已保存到: /fs-computility-new/nuclear/leishanzhe/repo/raycraft/output/")
    print()


# ============================================================================
# 示例 4: 自定义 action
# ============================================================================

def example_custom_actions():
    """使用不同的 action（使用YAML配置启用录制）"""
    print("\n=== 示例 4: 自定义 action（带视频录制）===")

    # 使用YAML配置启用录制
    yaml_config = "configs/simple_record.yaml"
    print(f"使用配置: {yaml_config}")

    env_id = create_env_ids(SERVER_URL, count=1, env_kwargs=yaml_config)[0]

    with RemoteEnv(SERVER_URL, env_id) as env:
        obs, info = env.reset()
        print("✓ Reset完成")

        # 测试LLM格式和Agent格式
        print("\n执行动作（混合使用LLM和Agent格式）:")

        # LLM格式的 action
        llm_actions = [
            '[{"action": "forward"}]',      # 前进
            '[{"action": "back"}]',         # 后退
            '[{"action": "left"}]',         # 左转
            '[{"action": "right"}]',        # 右转
            '[{"action": "jump"}]',         # 跳跃
            '[{"action": "attack"}]',       # 攻击
        ]

        # Agent格式的 action (buttons, camera索引)
        agent_actions = [
            {'buttons': 3, 'camera': 60},   # 前进
            {'buttons': 4, 'camera': 60},   # 后退
            {'buttons': 1, 'camera': 45},   # 左转
            {'buttons': 2, 'camera': 75},   # 右转
            {'buttons': 5, 'camera': 60},   # 跳跃
            {'buttons': 6, 'camera': 60},   # 攻击
        ]

        step_count = 0
        episode_done = False
        # 交替使用两种格式
        for llm_action, agent_action in zip(llm_actions, agent_actions):
            if episode_done:
                break

            # LLM格式
            for _ in range(8):
                obs, reward, term, trunc, info = env.step(llm_action)
                step_count += 1
                if step_count % 24 == 0:
                    print(f"  已执行 {step_count} 步")
                # if term or trunc:
                #     print(f"  ⚠ Episode terminated/truncated at step {step_count}")
                #     print(f"    terminated={term}, truncated={trunc}")
                #     print(f"    info keys: {list(info.keys())}")
                #     episode_done = True
                #     break

            if episode_done:
                break

            # Agent格式
            for _ in range(8):
                obs, reward, term, trunc, info = env.step(agent_action)
                step_count += 1
                if step_count % 24 == 0:
                    print(f"  已执行 {step_count} 步")
                # if term or trunc:
                    # episode_done = True
                    # break

        print(f"✓ 总共执行了 {step_count} 步")

        # 重要：再次reset以触发视频保存（RecordCallback在before_reset时保存）
        print("\n触发视频保存（调用reset）...")
        env.reset()
        print("✓ 视频已保存到: /fs-computility-new/nuclear/leishanzhe/repo/raycraft/output/")

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
# 示例 7: 使用 YAML 配置文件
# ============================================================================

def example_yaml_config():
    """使用 YAML 配置文件创建环境（推荐用于生产环境）"""
    print("\n=== 示例 7: 使用 YAML 配置文件 ===")

    # 使用 YAML 配置文件路径（相对于项目根目录）
    yaml_path = "configs/kill/kill_zombie_with_record.yaml"

    print(f"使用配置文件: {yaml_path}")

    # 方式1: 直接传递 YAML 路径字符串
    env_ids = create_env_ids(SERVER_URL, count=1, env_kwargs=yaml_path)
    env_id = env_ids[0]

    print(f"✓ 环境创建成功: {env_id}")

    with RemoteEnv(SERVER_URL, env_id) as env:
        obs, info = env.reset()
        print("✓ Reset完成")

        # 执行一些动作
        for i in range(10):
            obs, reward, term, trunc, info = env.step('[{"action": "attack"}]')
            print(f"  Step {i+1}: reward={reward:.2f}")

            # if term or trunc:
            #     break

    print("✓ 环境已关闭（视频应该已保存到 output/ 目录）")

    # 方式2: 传递配置字典（如果需要覆盖某些参数）
    print("\n方式2: 使用字典配置（可以自定义参数）")
    dict_config = {
        "resolution": [640, 360],
        "timestep_limit": 500
    }
    env_ids = create_env_ids(SERVER_URL, count=1, env_kwargs=dict_config)
    env_id = env_ids[0]
    print(f"✓ 环境创建成功（使用字典配置）: {env_id}")

    with RemoteEnv(SERVER_URL, env_id) as env:
        obs, info = env.reset()
        print("✓ Reset完成")

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
        # example_basic()
        # example_context_manager()
        example_batch_parallel()  # 这个比较慢，需要并行reset多个环境
        # example_custom_actions()  # 演示双action格式 + 视频录制
        # example_error_handling()
        # example_rgb_images()
        # example_yaml_config()  # 演示使用YAML配置文件

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
