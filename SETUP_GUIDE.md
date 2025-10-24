# Raycraft 设置指南

**状态：✅ 创建成功**

---

## 📦 已完成

### 1. Repo 结构创建 ✅

```
raycraft/
├── raycraft/                   # 主包（纯 Ray 实现）
│   ├── __init__.py             # 导出 MCRayClient, MCEnvActor
│   ├── ray/                    # Ray 模块
│   │   ├── client.py           # MCRayClient
│   │   ├── actors.py           # MCEnvActor
│   │   ├── pool.py             # EnvPool
│   │   └── global_pool.py      # 全局池
│   ├── mc_simulator.py         # 环境核心
│   └── utils/                  # 工具函数
│
├── MineStudio/                 # Git submodule（已复制）
├── configs/                    # YAML 配置（已复制）
├── examples/                   # 示例代码
│   ├── mvp1_basic_test.py      # ✅
│   ├── test_batch_create.py    # ✅
│   └── mvp1_migration_guide.py # ✅
├── docs/                       # 文档
│
├── pyproject.toml              # ✅ 项目配置
├── README.md                   # ✅ 项目说明
├── LICENSE                     # ✅ MIT
├── .gitignore                  # ✅
└── setup.sh                    # ✅ 安装脚本
```

### 2. 包名替换 ✅

- `agentenv_mc` → `raycraft` （全局替换完成）
- 验证：无遗留 `agentenv_mc` 引用

### 3. HTTP 部分清理 ✅

已删除：
- `envs/` - HTTP 客户端
- `server.py` - HTTP 服务器
- `launch.py` - 服务器启动
- `comp/` - HTTP 组件
- 旧版备份文件

### 4. 导入测试 ✅

```python
from raycraft import MCRayClient, MCEnvActor
# ✅ 导入成功
```

---

## 🚀 下一步操作

### 1. 初始化 Git Repo

```bash
cd /fs-computility-new/nuclear/leishanzhe/repo/AgentGym/raycraft

# 初始化 Git
git init

# 添加所有文件
git add .

# 首次提交
git commit -m "Initial commit: raycraft v1.0.0

- Pure Ray-based Minecraft Gym environment
- Extracted from AgentGym/agentenv-mc
- Zero HTTP overhead
- Parallel environment creation support
"
```

### 2. 配置 MineStudio Submodule

```bash
# 如果 MineStudio 是 Git submodule
cd /fs-computility-new/nuclear/leishanzhe/repo/AgentGym/raycraft

# 删除已复制的 MineStudio（准备用 submodule 替代）
rm -rf MineStudio

# 添加 submodule
git submodule add git@github.com:CraftJarvis/MineStudio.git MineStudio

# 提交 submodule 配置
git add .gitmodules MineStudio
git commit -m "Add MineStudio as submodule"
```

**或者保持直接复制**（如果不需要跟踪 MineStudio 更新）

### 3. 安装和测试

```bash
cd /fs-computility-new/nuclear/leishanzhe/repo/AgentGym/raycraft

# 运行安装脚本
bash setup.sh

# 或手动安装
pip install -e .

# 测试导入
python -c 'from raycraft import MCRayClient; print("✅ Import OK")'

# 运行示例（需要先确保环境正确）
python examples/mvp1_basic_test.py
```

### 4. 创建 GitHub Repo

```bash
# 创建远程 repo（在 GitHub 上）
# 然后关联本地 repo

cd /fs-computility-new/nuclear/leishanzhe/repo/AgentGym/raycraft

git remote add origin git@github.com:YOUR_ORG/raycraft.git
git branch -M main
git push -u origin main
```

### 5. 更新 README 中的链接

编辑 `README.md`，替换：
- `YOUR_ORG` → 你的 GitHub 组织/用户名

---

## 📝 使用示例

### 基础使用

```python
from raycraft import MCRayClient

# 批量创建环境
uuids = MCRayClient.create_batch(
    num_envs=10,
    config_path="configs/kill/kill_zombie_with_record.yaml"
)

# 连接环境
client = MCRayClient(uuid=uuids[0])

# 标准 Gym 接口
obs = client.reset()
for i in range(100):
    result = client.step('[{"action": "forward"}]')
    if result.done:
        break

client.close()
```

### 传统模式（单个环境）

```python
from raycraft import MCRayClient

# 直接创建（不使用 UUID 模式）
client = MCRayClient(config_path="configs/base.yaml")

obs = client.reset()
result = client.step('[{"action": "jump"}]')
client.close()
```

---

## 🔍 验证清单

在发布前，确保：

- [ ] ✅ Git repo 初始化
- [ ] ✅ MineStudio submodule 配置（或保持复制）
- [ ] ✅ `pip install -e .` 成功
- [ ] ✅ 导入测试通过
- [ ] ✅ 运行至少一个示例成功
- [ ] ✅ README 中的链接已更新
- [ ] ✅ 文档完整（docs/）
- [ ] ✅ LICENSE 正确

---

## 🐛 已知问题

### Issue 1: MineStudio 路径

`mc_simulator.py` 中硬编码了 MineStudio 路径：

```python
# 当前（第 6 行）
minestudio_path = Path(__file__).parent.parent / "MineStudio"
```

**解决方案：**
- 如果 MineStudio 安装到 Python 环境中，可以删除这段路径操作
- 或者保持当前方式（适用于开发模式）

### Issue 2: ToolBase 依赖

`mc_simulator.py` 中有简单的 ToolBase 替代品：

```python
class ToolBase:
    """简单的ToolBase替代品"""
    def __init__(self, name=None, **kwargs):
        self.name = name
```

**建议：**
- 完全移除 ToolBase，让 MCSimulator 不再继承任何基类
- 或者保持当前简单实现（兼容性考虑）

---

## 📚 相关资源

- **AgentGym 原始 Repo**: `/fs-computility-new/nuclear/leishanzhe/repo/AgentGym/agentenv-mc`
- **MineStudio**: https://github.com/CraftJarvis/MineStudio
- **Ray 文档**: https://docs.ray.io/

---

## 🎉 总结

**Raycraft 已成功创建！**

- ✅ 纯 Ray 架构，零 HTTP 开销
- ✅ 包名已统一为 `raycraft`
- ✅ HTTP 部分已清理
- ✅ 导入测试通过
- ✅ 项目配置完整

**保持简洁，拒绝过度设计！** 🚀
