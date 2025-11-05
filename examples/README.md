# RayCraft HTTP API 使用示例

## 快速开始

### 1. 启动服务器（在机器 A）

```bash
# 需要完整的 raycraft 环境
python -m raycraft.http_server
```

服务器将监听 `0.0.0.0:8000`

### 2. 运行客户端（在机器 B）

客户端**只需要 requests 库**，无需 ray、numpy、PIL 等重型依赖：

```bash
pip install requests
```

运行示例：

```bash
# 如果在同一台机器
python examples/http_client_example.py

# 如果在不同机器，修改 SERVER_URL
# 编辑 http_client_example.py:
#   SERVER_URL = "http://10.0.1.100:8000"
python examples/http_client_example.py
```

## 示例说明

### `http_client_example.py`

包含 6 个完整示例：

1. **基础使用** - 创建、reset、step、close
2. **上下文管理器** - 使用 `with` 自动管理生命周期（推荐）
3. **批量并行环境** - 创建多个环境并行运行（已注释，较慢）
4. **自定义 action** - 不同动作的使用
5. **错误处理** - 处理连接、超时等错误
6. **RGB 图像处理** - 如何处理压缩后的图像数据

## 最小示例

```python
from raycraft.http_client import create_remote_envs

# 创建环境
envs = create_remote_envs("http://localhost:8000", count=1)
env = envs[0]

# Reset
obs, info = env.reset()

# Step
for _ in range(10):
    obs, reward, terminated, truncated, info = env.step('[{"action": "forward"}]')
    if terminated or truncated:
        obs, info = env.reset()

# Close
env.close()
```

## 推荐用法（上下文管理器）

```python
from raycraft.http_client import create_remote_envs

envs = create_remote_envs("http://localhost:8000", count=1)

with envs[0] as env:
    obs, info = env.reset()
    for _ in range(10):
        obs, reward, term, trunc, info = env.step('[{"action": "forward"}]')
# 退出 with 后自动 close
```

## 批量并行训练

```python
from raycraft.http_client import create_remote_envs
from concurrent.futures import ThreadPoolExecutor

# 批量创建
envs = create_remote_envs("http://localhost:8000", count=8)

def train_episode(env):
    obs, info = env.reset()
    total_reward = 0
    for _ in range(100):
        obs, reward, term, trunc, info = env.step('[{"action": "forward"}]')
        total_reward += reward
        if term or trunc:
            break
    env.close()
    return total_reward

# 并行运行
with ThreadPoolExecutor(max_workers=8) as executor:
    rewards = list(executor.map(train_episode, envs))

print(f"平均奖励: {sum(rewards) / len(rewards):.2f}")
```

## Action 格式

RayCraft 使用 JSON 字符串格式的 action：

```python
# 基础动作
env.step('[{"action": "forward"}]')
env.step('[{"action": "back"}]')
env.step('[{"action": "left"}]')
env.step('[{"action": "right"}]')
env.step('[{"action": "jump"}]')
env.step('[{"action": "attack"}]')

# 组合动作（多个动作同时执行）
env.step('[{"action": "forward"}, {"action": "jump"}]')
```

详细的 action 格式见 [HTTP_API_GUIDE.md](../docs/HTTP_API_GUIDE.md)

## 性能注意事项

1. **首次 reset 慢** (~50秒)
   - Minecraft 需要启动和初始化世界
   - 后续 reset 使用 fast reset 机制，约 0.5 秒

2. **RGB 图像传输**
   - 服务器端自动压缩为 JPEG (quality=85)
   - 客户端收到 base64 编码的数据
   - 如需原始图像：
     ```python
     import base64, io
     from PIL import Image

     jpeg_bytes = base64.b64decode(obs['rgb']['data'])
     image = Image.open(io.BytesIO(jpeg_bytes))
     ```

3. **超时设置**
   - `reset(timeout=120)` - 首次 reset 建议 120 秒
   - `step(timeout=10)` - step 通常 < 1 秒
   - `close(timeout=30)` - close 需要保存 MP4，建议 30 秒

## 故障排查

### 连接错误

```
✗ 无法连接到服务器: http://localhost:8000
```

**解决方案**:
1. 确认服务器已启动：`python -m raycraft.http_server`
2. 检查防火墙：`sudo ufw allow 8000`
3. 检查地址：确保 IP 和端口正确

### 超时错误

```
ReadTimeout: HTTPConnectionPool(...): Read timed out. (read timeout=120)
```

**解决方案**:
- 增加超时时间：`env.reset(timeout=300)`
- 服务器资源不足，减少并行环境数量

### 序列化错误

```
PydanticSerializationError: Unable to serialize unknown type
```

**解决方案**:
- 这是服务器端问题，已在最新版本修复
- 更新到最新的 `raycraft/http_server.py`

## API 文档

完整 API 文档见：
- [HTTP API 使用指南](../docs/HTTP_API_GUIDE.md)
- [部署指南](../docs/DEPLOYMENT_GUIDE.md)
- [实现文档](../docs/IMPLEMENTATION.md)

## 更多示例

需要更多示例？提交 issue 或查看测试文件：
- `tests/test_http_client.py` - 客户端完整测试
- `tests/test_http_server.py` - 服务器端完整测试
