# RayCraft HTTP API

## 这是什么

把 Ray 环境池包装成 HTTP 服务，让训练代码可以跨机器访问 Minecraft 环境。

## 启动服务

```bash
python -m raycraft.http_server
```

默认监听 `0.0.0.0:8000`

## API

### 批量创建环境

```bash
curl -X POST http://server:8000/batch/envs \
  -H "Content-Type: application/json" \
  -d '{
    "count": 8,
    "env_name": "minecraft",
    "env_kwargs": {}
  }'
```

返回 `{"env_ids": ["uuid1", "uuid2", ...]}`

### Reset

```bash
curl -X POST http://server:8000/envs/{env_id}/reset
```

返回 `{"observation": {...}, "info": {...}}`

### Step

```bash
curl -X POST http://server:8000/envs/{env_id}/step \
  -H "Content-Type: application/json" \
  -d '{"action": "[{\"action\": \"forward\"}]"}'
```

**注意**：action是JSON字符串，不是数组！

返回 `{"observation": {...}, "reward": 0.0, "terminated": false, "truncated": false, "info": {...}}`

### 关闭

```bash
curl -X DELETE http://server:8000/envs/{env_id}
```

## Python 客户端（未实现）

```python
from raycraft.http_client import RemoteEnv
from concurrent.futures import ThreadPoolExecutor

# 批量创建
server = "http://server:8000"
resp = requests.post(f"{server}/batch/envs", json={"count": 8})
env_ids = resp.json()["env_ids"]

# 创建环境对象
envs = [RemoteEnv(server, env_id) for env_id in env_ids]

# 并行reset（训练代码自己控制并行）
with ThreadPoolExecutor(max_workers=8) as executor:
    obs_list = list(executor.map(lambda e: e.reset(), envs))

# 并行step
for step in range(1000):
    actions = ['[{"action": "forward"}]'] * 8

    def do_step(env, action):
        return env.step(action)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(do_step, envs, actions))

# 关闭
for env in envs:
    env.close()
```

## Action 格式

**正确**：JSON字符串
```python
action = '[{"action": "forward"}]'
action = '[{"action": "attack"}, {"action": "jump"}]'
```

**错误**：
```python
action = [0, 0, 0, 0, 0, 0, 0, 0]  # ❌
action = {"action": "forward"}      # ❌
```

## 注意事项

1. action是JSON字符串，与Ray保持一致
2. RGB图像自动压缩为JPEG (quality=85)
3. 批量创建环境，单独操作每个环境
4. 训练代码用ThreadPool自己控制并行
5. 必须调用close，否则资源泄漏

完了。
