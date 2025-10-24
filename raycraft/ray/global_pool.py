"""
全局环境池管理器
确保所有客户端共享同一个环境池实例
"""

import ray
from typing import Optional
from .pool import EnvPool

# 全局环境池引用
_global_env_pool: Optional[ray.ObjectRef] = None

def get_global_env_pool():
    """获取全局环境池实例

    Returns:
        Ray ObjectRef: 全局环境池的引用
    """
    global _global_env_pool

    if _global_env_pool is None:
        # 创建全局环境池
        _global_env_pool = EnvPool.remote()

    return _global_env_pool

def reset_global_env_pool():
    """重置全局环境池（用于测试）"""
    global _global_env_pool

    if _global_env_pool is not None:
        try:
            ray.kill(_global_env_pool)
        except:
            pass
        _global_env_pool = None

def clear_global_env_pool():
    """清理全局环境池（reset_global_env_pool的别名，用于测试）"""
    reset_global_env_pool()