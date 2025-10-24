"""
Raycraft - Pure Ray-based Minecraft Gym Environment

A lightweight, high-performance Minecraft RL environment powered by:
- Ray for distributed computing
- MineStudio for Minecraft simulation
- OpenAI Gym standard interface

Features:
- Parallel environment creation (10 envs in ~2 min)
- Automatic path isolation per environment
- Zero HTTP overhead (direct Ray RPC)
- Distributed training support
"""

from .ray.client import MCRayClient
from .ray.actors import MCEnvActor

__version__ = "1.0.0"
__all__ = ["MCRayClient", "MCEnvActor"]
