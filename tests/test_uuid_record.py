#!/usr/bin/env python3
"""
UUID录制测试 - 验证MP4是否保存到对应UUID目录
迁移自 AgentGym/agentenv-mc，适配 raycraft API
"""
import ray
import time
import cv2
import numpy as np
from pathlib import Path

# 使用相对路径导入 raycraft 模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from raycraft.ray.client import MCRayClient
from raycraft.ray.global_pool import get_global_env_pool

# 配置 - 使用相对路径
PROJECT_ROOT = Path(__file__).parent.parent
config_path = PROJECT_ROOT / "configs/kill/kill_zombie_with_record.yaml"
output_base = PROJECT_ROOT / "output"

print(f"📍 项目根目录: {PROJECT_ROOT}")
print(f"📍 配置文件: {config_path}")
print(f"📍 输出目录: {output_base}")

# 初始化Ray
print("\n🚀 初始化Ray...")
ray.init(ignore_reinit_error=True, log_to_driver=False, include_dashboard=False)
print("✅ Ray 初始化成功")

# 创建2个环境
print("\n📦 创建2个环境...")
try:
    uuids = MCRayClient.create_batch(num_envs=2, config_path=str(config_path))
    print(f"✅ 环境创建成功! UUID: {[u[:8] for u in uuids]}")
except Exception as e:
    print(f"❌ 环境创建失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 验证环境池中的环境
print("\n🔍 验证环境是否在环境池中...")
env_pool = get_global_env_pool()
for uuid in uuids:
    exists = ray.get(env_pool.env_exists.remote(uuid))
    print(f"   UUID {uuid[:8]}: {'✅ 存在' if exists else '❌ 不存在'}")

# 连接并运行
print("\n🎮 连接环境并执行操作...")
clients = []
for i, uuid in enumerate(uuids):
    print(f"\n环境 {i+1} (UUID: {uuid[:8]}):")
    uuid_short = uuid[:8]
    env_output_dir = output_base / f"env-{uuid_short}"
    env_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        client = MCRayClient(uuid=uuid)
        clients.append(client)
        print("   ✅ Client 创建成功")

        print("   🔄 Reset...")
        obs = client.reset()
        print(f"   ✅ Reset 成功 (obs type: {type(obs)})")

        # 保存 reset 的观察图像
        if isinstance(obs, dict) and 'image' in obs:
            reset_img_path = env_output_dir / "reset_obs.png"
            cv2.imwrite(str(reset_img_path), cv2.cvtColor(obs['image'], cv2.COLOR_RGB2BGR))
            print(f"   💾 保存 reset 图像: {reset_img_path}")

        print("   ▶️  执行10步...")
        for step in range(10):
            result = client.step('[{"action": "forward"}]')

            # 保存每一步的观察图像
            if isinstance(result.observation, dict) and 'image' in result.observation:
                step_img_path = env_output_dir / f"step_{step:03d}_obs.png"
                cv2.imwrite(str(step_img_path), cv2.cvtColor(result.observation['image'], cv2.COLOR_RGB2BGR))
                if step == 0:
                    print(f"   ✅ Step 成功 (reward: {result.reward}, done: {result.done})")
                    print(f"   💾 保存图像到: {env_output_dir}/step_XXX_obs.png")

        print("   🔒 Close...")
        client.close()
        print("   ✅ 完成")
    except Exception as e:
        print(f"   ❌ 错误: {e}")
        import traceback
        traceback.print_exc()

print("\n⏳ 等待1秒...")
time.sleep(1)

# 检查结果
print("\n📂 检查结果:")
for uuid in uuids:
    uuid_short = uuid[:8]
    expected_dir = output_base / f"env-{uuid_short}"

    if not expected_dir.exists():
        print(f"❌ env-{uuid_short}/: 目录不存在")
        continue

    # 检查图像文件
    png_files = list(expected_dir.glob("*.png"))
    mp4_files = list(expected_dir.glob("*.mp4"))

    print(f"\n📁 env-{uuid_short}/:")
    print(f"   🖼️  PNG 图像: {len(png_files)} 个")
    if png_files:
        for png in sorted(png_files)[:3]:  # 显示前3个
            print(f"      - {png.name}")
        if len(png_files) > 3:
            print(f"      ... 还有 {len(png_files) - 3} 个")

    print(f"   🎬 MP4 视频: {len(mp4_files)} 个")
    if mp4_files:
        for mp4 in mp4_files:
            print(f"      - {mp4.name}")

    # 判断
    if mp4_files:
        print(f"   ✅ 录制成功!")
    else:
        print(f"   ⚠️  无MP4文件 (但有 {len(png_files)} 个PNG)")

print("\n" + "="*60)
total_envs = len(uuids)
success_envs = sum(1 for u in uuids if len(list((output_base / f"env-{u[:8]}").glob("*.mp4"))) > 0)
print(f"✅ 测试完成: {success_envs}/{total_envs} 个环境成功录制MP4" if success_envs == total_envs else f"⚠️  部分成功: {success_envs}/{total_envs} 个环境录制MP4")

# 清理Ray
ray.shutdown()
