# AgentEnv-MC Pure Ray 架构设计

## 核心哲学

**"好品味"：消除所有特殊情况，数据结构优先，原生分布式。**

## 设计动机

### 现有架构问题
- **HTTP开销**：序列化/反序列化浪费CPU和内存
- **单点瓶颈**：FastAPI服务器成为性能瓶颈
- **复杂部署**：需要维护HTTP服务器 + 环境管理逻辑

### Pure Ray 优势
- **原生分布式**：天然支持跨机器通信，零配置
- **零拷贝传输**：大对象通过共享内存传输
- **统一协议**：训练和环境使用相同的Ray协议栈
- **自动负载均衡**：Ray自动将Actor分布到最优节点

## 架构设计

### 核心数据流
```
训练进程 ──ray协议──→ Ray集群 ──→ MCEnvActor ──→ MCSimulator
    ↑                      ↓
    └─── Ray Object Store ──┘
        (零拷贝观察数据)
```

### 组件设计

#### 1. MCEnvActor
**职责**：封装单个MC环境实例，提供Ray原生接口
```python
import ray
from verl.workers.agent.envs.mc.mc_simulator import MCSimulator

@ray.remote
class MCEnvActor:
    def __init__(self, config: dict):
        """初始化MC环境
        Args:
            config: 环境配置，包含分辨率、生物群系等参数
        """
        self.simulator = MCSimulator(config)
        self.config = config
        self.step_count = 0

    def reset(self):
        """重置环境，返回初始观察"""
        obs = self.simulator.reset()
        self.step_count = 0
        # 大对象自动存储到Object Store
        return ray.put(obs) if self._is_large_object(obs) else obs

    def step(self, action: str):
        """执行动作
        Args:
            action: JSON格式动作字符串
        Returns:
            (observation, reward, done, info)
        """
        obs, reward, done, info = self.simulator.step(action)
        self.step_count += 1

        # 大对象优化
        obs_ref = ray.put(obs) if self._is_large_object(obs) else obs

        return obs_ref, reward, done, {
            **info,
            "step_count": self.step_count,
            "actor_id": ray.get_runtime_context().actor_id
        }

    def get_observation(self):
        """获取当前观察"""
        obs = self.simulator.get_observation()
        return ray.put(obs) if self._is_large_object(obs) else obs

    def get_stats(self):
        """获取环境统计信息"""
        return {
            "step_count": self.step_count,
            "config": self.config,
            "actor_id": ray.get_runtime_context().actor_id,
            "node_id": ray.get_runtime_context().node_id
        }

    def _is_large_object(self, obj):
        """判断是否为大对象（需要存储到Object Store）"""
        import sys
        return sys.getsizeof(obj) > 1024 * 1024  # 1MB阈值
```

#### 2. EnvPool
**职责**：环境池管理，支持动态扩展和负载均衡
```python
@ray.remote
class EnvPool:
    def __init__(self, pool_size: int = 10):
        """初始化环境池
        Args:
            pool_size: 预创建的环境数量
        """
        self.pool_size = pool_size
        self.env_configs = {}
        self.available_envs = []
        self.busy_envs = {}

    def create_env(self, config: dict) -> ray.ObjectRef:
        """创建新环境实例
        Args:
            config: 环境配置
        Returns:
            环境Actor的引用
        """
        env_actor = MCEnvActor.remote(config)
        env_id = ray.get_object_ref_id(env_actor)
        self.env_configs[env_id] = config
        return env_actor

    def get_env(self, config: dict = None):
        """获取可用环境（复用或新建）
        Args:
            config: 环境配置，如果为None则使用默认配置
        Returns:
            环境Actor引用
        """
        if self.available_envs:
            return self.available_envs.pop()
        else:
            return self.create_env(config or self._get_default_config())

    def return_env(self, env_actor):
        """归还环境到池中"""
        self.available_envs.append(env_actor)

    def _get_default_config(self):
        """获取默认环境配置"""
        return {
            "resolution": [640, 360],
            "preferred_spawn_biome": "plains",
            "action_type": "agent",
            "timestep_limit": 1000
        }
```

#### 3. MCRayClient
**职责**：Ray原生客户端，替代HTTP客户端
```python
import ray

class MCRayClient:
    def __init__(self, ray_address: str = None, config: dict = None):
        """初始化Ray客户端
        Args:
            ray_address: Ray集群地址，如 'ray://localhost:10001'
            config: 环境配置
        """
        # 连接到Ray集群
        if ray_address:
            ray.init(address=ray_address)
        elif not ray.is_initialized():
            ray.init()

        self.config = config or self._get_default_config()
        self.env_actor = None

    def create(self):
        """创建环境实例"""
        self.env_actor = MCEnvActor.remote(self.config)
        return self.reset()

    def reset(self):
        """重置环境"""
        if not self.env_actor:
            self.create()
        obs_ref = ray.get(self.env_actor.reset.remote())
        return self._resolve_object(obs_ref)

    def step(self, action: str):
        """执行动作"""
        if not self.env_actor:
            raise RuntimeError("Environment not created. Call create() first.")

        result = ray.get(self.env_actor.step.remote(action))
        obs_ref, reward, done, info = result

        return StepResult(
            observation=self._resolve_object(obs_ref),
            reward=reward,
            done=done,
            info=info
        )

    def observe(self):
        """获取当前观察"""
        if not self.env_actor:
            raise RuntimeError("Environment not created. Call create() first.")
        obs_ref = ray.get(self.env_actor.get_observation.remote())
        return self._resolve_object(obs_ref)

    def close(self):
        """关闭环境"""
        if self.env_actor:
            ray.kill(self.env_actor)
            self.env_actor = None

    def _resolve_object(self, obj_or_ref):
        """解析对象引用"""
        if isinstance(obj_or_ref, ray.ObjectRef):
            return ray.get(obj_or_ref)
        return obj_or_ref

    def _get_default_config(self):
        return {
            "resolution": [640, 360],
            "preferred_spawn_biome": "plains",
            "action_type": "agent",
            "timestep_limit": 1000
        }

class StepResult:
    """步骤结果封装"""
    def __init__(self, observation, reward, done, info):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info
```

## 部署架构

### 单机部署
```bash
# 启动Ray集群（本地模式）
ray start --head --dashboard-host=0.0.0.0

# 训练代码
python train.py  # 自动连接本地Ray
```

### 跨机器部署
```bash
# 机器B：环境服务器
ray start --head --dashboard-host=0.0.0.0 --port=6379 --dashboard-port=8265

# 机器A：训练代码
python train.py --ray-address="ray://机器B:10001"
```

### 集群部署
```bash
# 头节点
ray start --head --dashboard-host=0.0.0.0 --port=6379

# 工作节点1
ray start --address='头节点IP:6379'

# 工作节点2
ray start --address='头节点IP:6379'

# 训练代码（可在任意机器）
python train.py --ray-address="ray://头节点IP:10001"
```

## 使用方式

### 基础使用
```python
# 训练代码中
import ray
from raycraft import MCRayClient

# 连接到远程Ray集群
client = MCRayClient(
    ray_address="ray://env-server:10001",
    config={
        "resolution": [640, 360],
        "preferred_spawn_biome": "plains"
    }
)

# 创建环境
obs = client.create()

# 训练循环
for episode in range(1000):
    obs = client.reset()
    total_reward = 0

    for step in range(500):
        action = policy.predict(obs)
        result = client.step(action)

        obs = result.observation
        total_reward += result.reward

        if result.done:
            break

    print(f"Episode {episode}: Reward = {total_reward}")

client.close()
```

### 批量环境使用
```python
import ray
from raycraft import MCEnvActor

# 连接集群
ray.init(address="ray://env-server:10001")

# 创建多个环境
num_envs = 20
envs = [MCEnvActor.remote(config) for _ in range(num_envs)]

# 并行重置
reset_futures = [env.reset.remote() for env in envs]
observations = ray.get(reset_futures)

# 并行执行动作
actions = [generate_action(obs) for obs in observations]
step_futures = [env.step.remote(action) for env, action in zip(envs, actions)]
results = ray.get(step_futures)

# 处理结果
for i, (obs, reward, done, info) in enumerate(results):
    print(f"Env {i}: Reward = {reward}")
```

### 配置文件支持
```python
# config.yaml
ray:
  address: "ray://env-server:10001"

environment:
  resolution: [640, 360]
  preferred_spawn_biome: "plains"
  timestep_limit: 1000

training:
  num_envs: 16
  max_episodes: 1000

# 训练代码
import yaml
from raycraft import MCRayClient

with open('config.yaml') as f:
    config = yaml.safe_load(f)

# 自动配置Ray连接
client = MCRayClient(
    ray_address=config['ray']['address'],
    config=config['environment']
)
```

## 性能优化

### 1. 大对象优化
- **自动检测**：超过1MB的观察数据自动存储到Object Store
- **零拷贝**：图像数据通过共享内存传输
- **压缩传输**：可选的观察数据压缩

### 2. 并发优化
- **批量操作**：支持多环境并行执行
- **异步调用**：使用`remote()`进行非阻塞调用
- **流水线**：重叠计算和通信

### 3. 资源管理
- **动态扩展**：根据负载自动创建/销毁Actor
- **节点亲和性**：GPU环境优先调度到GPU节点
- **内存管理**：自动垃圾回收和内存压缩

## 与现有代码的迁移

### 从HTTP客户端迁移
```python
# 原来的代码
from raycraft import MCEnvClient

client = MCEnvClient(
    env_server_base="http://localhost:8000",
    data_len=10,
    resolution=[640, 360]
)

# 迁移后的代码
from raycraft import MCRayClient

client = MCRayClient(
    ray_address="ray://localhost:10001",  # 唯一变化
    config={
        "resolution": [640, 360],
        "timestep_limit": 10
    }
)

# 其他API完全相同
obs = client.create()
result = client.step(action)
client.close()
```

### 渐进迁移策略
1. **第一阶段**：部署Ray集群，HTTP服务继续运行
2. **第二阶段**：部分训练任务切换到Ray客户端
3. **第三阶段**：性能验证后，全面切换到Ray
4. **第四阶段**：移除HTTP服务器代码

## 实施计划

### 阶段1：核心Actor实现 (2天)
- [ ] 实现 MCEnvActor，包装 MCSimulator
- [ ] 实现 MCRayClient，提供原生Ray接口
- [ ] 基础测试：单环境创建、步骤执行

### 阶段2：并发和优化 (3天)
- [ ] 实现 EnvPool 环境池管理
- [ ] 大对象优化：Object Store集成
- [ ] 并发测试：多环境并行执行

### 阶段3：生产特性 (2天)
- [ ] 配置文件支持
- [ ] 错误处理和重连机制
- [ ] 性能监控和指标收集

### 阶段4：集成测试 (1天)
- [ ] 端到端集成测试
- [ ] 性能基准测试
- [ ] 文档和示例代码

## 性能预期

### 延迟优化
- **本地部署**：比HTTP减少50%+延迟（去除序列化开销）
- **跨机器**：比HTTP减少30%+延迟（优化的网络协议）

### 吞吐量提升
- **单机**：支持100+并发环境（vs HTTP的20+）
- **集群**：线性扩展，理论无上限

### 内存效率
- **零拷贝**：大对象传输内存使用减少70%+
- **共享内存**：多环境共享相同观察数据

## 总结

Pure Ray架构彻底简化了agentenv-mc的设计：

1. **消除HTTP层**：直接使用Ray的原生分布式通信
2. **统一协议栈**：训练和环境使用相同的Ray基础设施
3. **原生分布式**：天然支持跨机器和集群部署
4. **零配置扩展**：Ray自动处理负载均衡和故障恢复

这是真正的"好品味"设计：**用正确的工具解决正确的问题**。