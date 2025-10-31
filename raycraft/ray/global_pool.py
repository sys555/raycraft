"""
全局环境池管理器
确保所有客户端共享同一个环境池实例（跨进程）
"""

import ray
from typing import Optional
from .pool import EnvPool

# 全局环境池的命名
GLOBAL_POOL_NAME = "GlobalEnvPool"

# 本地缓存（进程级优化）
_global_env_pool: Optional[ray.ObjectRef] = None

def get_global_env_pool():
    """获取全局环境池实例（跨进程共享）

    使用 Ray Named Actor 确保所有进程（主进程和 Worker）
    访问同一个 EnvPool 实例

    Returns:
        Ray Actor Handle: 全局环境池的引用
    """
    global _global_env_pool

    # 优化：如果本地已缓存，直接返回
    if _global_env_pool is not None:
        return _global_env_pool

    try:
        # 尝试获取已存在的命名 Actor
        _global_env_pool = ray.get_actor(GLOBAL_POOL_NAME)
    except ValueError:
        # Actor 不存在，创建新的命名 Actor
        # get_if_exists=False 确保只有第一个调用者创建
        _global_env_pool = EnvPool.options(
            name=GLOBAL_POOL_NAME,
            lifetime="detached",  # 即使创建者退出也保持存活
            max_concurrency=100   # 支持高并发
        ).remote()

    return _global_env_pool

def reset_global_env_pool():
    """重置全局环境池（用于测试）

    清理命名 Actor 和本地缓存
    """
    global _global_env_pool

    # 清理命名 Actor
    try:
        actor = ray.get_actor(GLOBAL_POOL_NAME)
        ray.kill(actor)
    except ValueError:
        # Actor 不存在，忽略
        pass
    except Exception as e:
        # 其他错误，记录但继续
        print(f"Warning: Failed to kill global pool: {e}")

    # 清理本地缓存
    _global_env_pool = None

def clear_global_env_pool():
    """清理全局环境池（reset_global_env_pool的别名，用于测试）"""
    reset_global_env_pool()