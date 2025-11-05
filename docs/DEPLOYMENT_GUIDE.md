# RayCraft 部署指南

## 服务端（环境节点）

### 1. 安装依赖

```bash
cd /path/to/raycraft
pip install -r requirements.txt
pip install fastapi uvicorn httpx
```

### 2. 启动服务

**开发环境：**
```bash
python -m raycraft.http_server
```

**rjob 环境：**
```bash
rjob submit \
    --name raycraft-server \
    --cpu 16 \
    --memory 32G \
    --command "cd /path/to/raycraft && python -m raycraft.http_server"
```

### 3. 验证

```bash
curl http://localhost:8000/health
```

应该返回 `{"status": "healthy", ...}`

## 客户端（训练节点）

### 1. 安装依赖

```bash
pip install requests numpy pillow
```

### 2. 复制客户端文件

```bash
# 从服务端复制
scp server:/path/to/raycraft/raycraft/http_client.py ./
```

### 3. 使用

```python
from http_client import RemoteEnv

env = RemoteEnv("http://server-ip:8000")
obs, info = env.reset()
obs, reward, done, _, info = env.step(action)
env.close()
```

## 网络要求

- 训练节点能访问环境节点的 8000 端口
- 建议千兆网络，延迟 < 10ms

## 故障排查

**连接失败：**
```bash
# 检查服务是否运行
ps aux | grep http_server

# 检查端口
netstat -tulnp | grep 8000

# 测试连接
curl http://server-ip:8000/health
```

**服务崩溃：**
```bash
# 查看日志
rjob logs raycraft-server

# 重启服务
rjob kill raycraft-server
rjob submit ...
```

完了。
