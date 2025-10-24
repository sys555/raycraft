# AgentEnv-MC 架构文档

## 项目概述

AgentEnv-MC 是基于 MineStudio/DeepEyes 的 Minecraft 强化学习环境，支持两种部署模式。

---

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户应用层                            │
│                                                             │
│  from raycraft.ray import MCRayClient                    │
│                                                             │
│  # 批量创建                                                  │
│  uuids = MCRayClient.create_batch(10, config_path=...)      │
│                                                             │
│  # 单独使用                                                  │
│  client = MCRayClient(uuid=uuids[0])                        │
│  obs = client.reset()                                       │
│  result = client.step(action)                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ Ray RPC
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Ray 分布式层                            │
│                                                             │
│  ┌──────────────────────────────────────────┐              │
│  │  Global EnvPool (@ray.remote)            │              │
│  │  管理: UUID → MCEnvActor 映射             │              │
│  └──────────────────┬───────────────────────┘              │
│                     │                                       │
│         ┌───────────┼───────────┐                          │
│         ▼           ▼           ▼                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│  │MCEnvActor│ │MCEnvActor│ │MCEnvActor│                   │
│  │@ray.remote│ │@ray.remote│ │@ray.remote│                   │
│  └─────┬────┘ └─────┬────┘ └─────┬────┘                   │
└────────┼────────────┼────────────┼─────────────────────────┘
         │            │            │
         ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────┐
│                    环境实现层                                │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│  │MCSimulator│  │MCSimulator│  │MCSimulator│              │
│  │   ↓       │  │   ↓       │  │   ↓       │               │
│  │MinecraftSim│ │MinecraftSim│ │MinecraftSim│             │
│  │(MineStudio)│ │(MineStudio)│ │(MineStudio)│             │
│  └──────────┘   └──────────┘   └──────────┘               │
│                                                             │
│  录制输出:                                                   │
│  output/env-{UUID}/episode_0.mp4                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心组件

### 1. MCRayClient (client.py)
**用户接口层**

- `create_batch(num_envs, config_path)` - 批量创建环境
- `__init__(uuid)` - 连接已有环境
- `reset() / step() / close()` - 标准 Gym 接口

### 2. EnvPool (pool.py)
**环境池管理**

- UUID → MCEnvActor 映射
- 并行环境创建
- 环境状态管理

### 3. MCEnvActor (actors.py)
**Ray Actor 封装**

- 独立进程隔离
- 包装 MCSimulator
- 自动故障恢复

### 4. MCSimulator (mc_simulator.py)
**环境适配层**

- 适配 AgentGym 接口
- 管理 MinecraftSim 生命周期
- 处理观察和奖励

---

## 两种部署模式

### HTTP 模式（传统）
```python
from raycraft import MCEnvClient

client = MCEnvClient(env_server_base="http://localhost:8000")
```

- 需要启动 HTTP 服务器
- 单机部署
- 有序列化开销

### Ray 模式（推荐）
```python
from raycraft.ray import MCRayClient

# 批量创建
uuids = MCRayClient.create_batch(10, config_path="...")

# 单独使用
client = MCRayClient(uuid=uuids[0])
```

- 无需 HTTP 服务器
- 原生分布式
- 并行创建
- 路径自动隔离

---

## 关键特性

### 1. UUID-based 环境池
```python
uuids = MCRayClient.create_batch(10, config_path="...")
# 返回: ['abc12345...', 'def67890...', ...]

client = MCRayClient(uuid=uuids[0])
```

### 2. 并行环境创建
```python
# 10个环境并行初始化（~2分钟）
# vs 串行创建（~20分钟）
uuids = MCRayClient.create_batch(10, config_path="...")
```

### 3. 自动路径隔离
```
output/
├── env-abc12345/
│   └── episode_0.mp4
└── env-def67890/
    └── episode_0.mp4
```

### 4. 符合 Gym 标准
```python
obs = client.reset()  # 直接返回观察
result = client.step(action)
# result.observation, result.reward, result.done
```

---

## 文件结构

```
agentenv-mc/
├── raycraft/
│   ├── ray/              # Ray 模式实现
│   │   ├── client.py     # MCRayClient
│   │   ├── actors.py     # MCEnvActor
│   │   ├── pool.py       # EnvPool
│   │   └── global_pool.py
│   ├── envs/             # HTTP 模式
│   │   └── minecraft.py  # MCEnvClient
│   └── mc_simulator.py   # 环境核心
├── configs/              # 配置文件
├── examples/             # 使用示例
├── tests/                # 测试
└── docs/                 # 文档
```

---

## 使用示例

### 基础使用
```python
from raycraft.ray import MCRayClient

# 创建环境
uuids = MCRayClient.create_batch(5, config_path="configs/kill/...")
client = MCRayClient(uuid=uuids[0])

# 使用 Gym 标准接口
obs = client.reset()
for _ in range(100):
    result = client.step('[{"action": "forward"}]')
    if result.done:
        obs = client.reset()

client.close()
```

### 并行使用
```python
# 创建多个环境
uuids = MCRayClient.create_batch(10, config_path="...")

# 并行操作
clients = [MCRayClient(uuid=u) for u in uuids]
for client in clients:
    client.reset()
    # 各自独立运行
```

---

## 设计原则

### 1. 数据结构优先
- UUID → Actor 清晰映射
- 全局单例环境池

### 2. 消除特殊情况
- 自动路径修改
- 统一配置处理

### 3. 向后兼容
- 保留 HTTP 模式
- 保留 MVP1 接口

### 4. 实用主义
- 解决真实问题（并行创建、路径隔离）
- 不过度设计
