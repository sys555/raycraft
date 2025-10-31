"""
MCEnvActor - Ray Actor封装MCSimulator
直接复用现有的MCSimulator，零修改
"""

import ray
import sys
import time
import logging
import os
from datetime import datetime
from pathlib import Path

# 配置日志保存到文件
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"actor_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.txt"

logger = logging.getLogger("MCEnvActor")
logger.setLevel(logging.INFO)

# 添加文件处理器
fh = logging.FileHandler(log_file, encoding="utf-8")
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# 同时保留控制台输出
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info(f"MCEnvActor logging to file: {log_file.absolute()}")

@ray.remote(max_restarts=3)
class MCEnvActor:
    """Ray Actor封装的MC环境实例"""

    def __init__(self, config: dict):
        """初始化MC环境

        Args:
            config: 环境配置，与HTTP版本完全兼容
        """
        try:
            # 确保项目根目录在sys.path中（Ray Actor在独立进程中运行）
            project_root = Path(__file__).parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            # 确保MineStudio也在sys.path中
            minestudio_path = project_root / "MineStudio"
            if minestudio_path.exists() and str(minestudio_path) not in sys.path:
                sys.path.insert(0, str(minestudio_path))
            
            # 直接复用现有MCSimulator
            from ..mc_simulator import MCSimulator

            self.config = config

            # Debug: 打印 callback 信息
            if isinstance(config, dict) and 'sim_callbacks' in config:
                logger.info(f"MCEnvActor: Received {len(config['sim_callbacks'])} callbacks")
                for i, cb in enumerate(config['sim_callbacks']):
                    cb_name = type(cb).__name__
                    if hasattr(cb, 'record_path'):
                        logger.info(f"  Callback {i}: {cb_name}, record_path={cb.record_path}")
                    else:
                        logger.info(f"  Callback {i}: {cb_name}")

            # 传递 config 到 MCSimulator.__init__, MinecraftSim 在 __init__ 时创建
            self.simulator = MCSimulator(config=config)
            self.step_count = 0
            self.created_at = time.time()

            logger.info(f"MCEnvActor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MCEnvActor: {e}")
            raise

    def reset(self):
        """重置环境

        Returns:
            观察数据（与HTTP版本格式一致）
        """
        try:
            # MinecraftSim 已在 __init__ 中创建,reset 只需调用 self.simulator.reset()
            obs, _info = self.simulator.reset()
            self.step_count = 0
            logger.debug("Environment reset successfully")

            # 大对象优化：自动检测并存储到Object Store
            return self._optimize_large_object(obs)

        except Exception as e:
            logger.error(f"Reset failed: {e}")
            raise

    def step(self, action: str):
        """执行动作

        Args:
            action: JSON格式的动作字符串（与HTTP版本一致）

        Returns:
            (observation, reward, done, info) - 与HTTP版本格式一致
        """
        try:
            obs, reward, done, info = self.simulator.step(action)
            self.step_count += 1

            logger.debug(f"Step {self.step_count}: reward={reward}, done={done}")

            # 增强info信息
            if info is None:
                info = {}
            info.update({
                "step_count": self.step_count,
                "actor_id": ray.get_runtime_context().actor_id,
                "node_id": ray.get_runtime_context().node_id
            })

            # 大对象优化
            obs_optimized = self._optimize_large_object(obs)

            return obs_optimized, reward, done, info

        except Exception as e:
            logger.error(f"Step failed: {e}")
            raise

    def get_observation(self):
        """获取当前观察

        Returns:
            当前观察数据
        """
        try:
            obs = self.simulator.get_observation()
            return self._optimize_large_object(obs)

        except Exception as e:
            logger.error(f"Get observation failed: {e}")
            raise

    def get_stats(self):
        """获取环境统计信息

        Returns:
            统计信息字典
        """
        return {
            "step_count": self.step_count,
            "config": self.config,
            "created_at": self.created_at,
            "uptime": time.time() - self.created_at,
            "actor_id": ray.get_runtime_context().actor_id,
            "node_id": ray.get_runtime_context().node_id
        }

    def _optimize_large_object(self, obj):
        """大对象优化：超过1MB自动存储到Object Store

        Args:
            obj: 要优化的对象

        Returns:
            优化后的对象或ObjectRef
        """
        try:
            # 检查对象大小
            if isinstance(obj, (str, bytes)) and sys.getsizeof(obj) > 1024 * 1024:  # 1MB
                logger.debug(f"Large object detected ({sys.getsizeof(obj)} bytes), storing to Object Store")
                return ray.put(obj)
            return obj

        except Exception as e:
            logger.warning(f"Large object optimization failed: {e}")
            return obj

    def close(self):
        """关闭环境（清理资源）"""
        try:
            # 需要调用内层的 MinecraftSim.close() 来触发 RecordCallback 保存 MP4
            if hasattr(self.simulator, 'simulator') and hasattr(self.simulator.simulator, 'close'):
                logger.info("Calling simulator.simulator.close() to trigger recording save")
                self.simulator.simulator.close()
            elif hasattr(self.simulator, 'close'):
                logger.info("Calling simulator.close()")
                self.simulator.close()

            logger.info("Environment closed successfully")

        except Exception as e:
            logger.error(f"Close failed: {e}")
            raise