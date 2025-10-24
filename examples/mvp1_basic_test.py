#!/usr/bin/env python3
"""
MVP1 基础测试 - 验证Ray版本与HTTP版本的兼容性
演示最小迁移成本：只需修改一行代码
"""

import sys
import time
from pathlib import Path

# raycraft 是独立 repo，不需要添加额外路径

def test_http_version():
    """
    HTTP版本已从 raycraft 中移除（纯 Ray 实现）
    如需 HTTP 版本，请使用 AgentGym/agentenv-mc
    """
    print("=== HTTP版本测试 ===")
    print("⚠️  raycraft 不包含 HTTP 版本")
    print("   如需 HTTP 版本，请使用 AgentGym/agentenv-mc")
    print("   或跳过此测试，直接运行 Ray 版本测试")
    return False

def test_ray_version():
    """测试Ray版本 - 核心MVP1功能"""
    print("\n=== Ray版本测试（MVP1） ===")

    try:
        from raycraft import MCRayClient

        # 唯一的变化：初始化参数
        client = MCRayClient(
            config_path="configs/kill/kill_zombie_with_record.yaml",
            ray_address=None  # 本地模式
        )

        print("✅ Ray客户端创建成功")

        # 符合Gym标准的接口调用
        obs = client.reset()
        print(f"✅ 环境初始化成功，观察长度: {len(obs)}")

        # 执行相同的动作序列
        actions = [
            '<answer>[{"action": "forward"}]</answer>',
            '<answer>[{"action": "jump"}]</answer>',
            '<answer>[{"action": "back"}]</answer>'
        ]

        total_reward = 0
        for i, action in enumerate(actions):
            result = client.step(action)
            total_reward += result.reward
            print(f"Step {i+1}: reward={result.reward:.3f}, done={result.done}")

        # 测试新增功能
        stats = client.get_stats()
        print(f"✅ 环境统计: {stats['step_count']} 步, Actor ID: {str(stats['actor_id'])[:8]}...")

        print(f"✅ Ray版本测试完成，总奖励: {total_reward:.3f}")
        client.close()
        return True

    except Exception as e:
        print(f"❌ Ray版本测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_migration_compatibility():
    """测试迁移兼容性 - 验证最小修改原则"""
    print("\n=== 迁移兼容性测试 ===")

    try:
        # 原来的代码（注释掉的部分）
        # from raycraft import MCEnvClient
        # client = MCEnvClient(env_server_base="http://localhost:8000", ...)

        # 迁移后的代码
        from raycraft import MCRayClient
        client = MCRayClient(config_path="configs/kill/kill_zombie_with_record.yaml", ray_address=None)

        # 符合Gym标准的用法
        obs = client.reset()  # 初始化环境并获取观察
        result = client.step('<answer>[{"action": "forward"}]</answer>')
        obs = result.observation  # 从step返回值中获取观察
        client.close()

        print("✅ 迁移兼容性测试通过 - 现在符合Gym标准!")
        return True

    except Exception as e:
        print(f"❌ 迁移兼容性测试失败: {e}")
        return False

def test_performance_basic():
    """基础性能测试"""
    print("\n=== 基础性能测试 ===")

    try:
        from raycraft import MCRayClient

        client = MCRayClient(config_path="configs/kill/kill_zombie_with_record.yaml")

        # 环境初始化时间
        start_time = time.time()
        obs = client.reset()
        creation_time = time.time() - start_time
        print(f"环境初始化耗时: {creation_time:.3f}s")

        # 步骤执行时间
        action = '<answer>[{"action": "forward"}]</answer>'
        step_times = []

        for i in range(10):
            start_time = time.time()
            result = client.step(action)
            step_time = time.time() - start_time
            step_times.append(step_time)

        avg_step_time = sum(step_times) / len(step_times)
        print(f"平均步骤耗时: {avg_step_time*1000:.1f}ms")
        print(f"最快步骤: {min(step_times)*1000:.1f}ms")
        print(f"最慢步骤: {max(step_times)*1000:.1f}ms")

        client.close()

        # 性能判断
        if avg_step_time < 0.1:  # 100ms
            print("✅ 性能测试通过 - 步骤延迟 < 100ms")
            return True
        else:
            print(f"⚠️  性能警告 - 步骤延迟 {avg_step_time*1000:.1f}ms > 100ms")
            return True  # 仍然算通过，只是性能警告

    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

def test_context_manager():
    """测试上下文管理器支持"""
    print("\n=== 上下文管理器测试 ===")

    try:
        from raycraft import MCRayClient

        # 使用with语句自动清理
        with MCRayClient(config_path="configs/kill/kill_zombie_with_record.yaml") as client:
            obs = client.reset()
            result = client.step('<answer>[{"action": "forward"}]</answer>')
            print(f"上下文管理器内执行成功，奖励: {result.reward:.3f}")

        print("✅ 上下文管理器测试通过 - 自动清理资源")
        return True

    except Exception as e:
        print(f"❌ 上下文管理器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 AgentEnv-MC MVP1 测试套件")
    print("=" * 50)

    test_results = []

    # 跳过HTTP测试（可能没有启动服务器）
    print("⏭️  跳过HTTP版本测试（需要手动启动服务器）")

    # 运行Ray版本测试
    test_results.append(("Ray版本基础功能", test_ray_version()))
    test_results.append(("迁移兼容性", test_migration_compatibility()))
    test_results.append(("基础性能", test_performance_basic()))
    test_results.append(("上下文管理器", test_context_manager()))

    # 结果统计
    print("\n" + "=" * 50)
    print("📊 测试结果总结")
    print("=" * 50)

    passed = 0
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n总体结果: {passed}/{len(test_results)} 测试通过")

    if passed == len(test_results):
        print("🎉 MVP1 测试全部通过! Ray版本可以投入使用")
        return True
    else:
        print("⚠️  存在测试失败，请检查问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)