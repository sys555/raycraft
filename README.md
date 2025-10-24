# Raycraft

**Pure Ray-based Minecraft Gym Environment**

A lightweight, high-performance Minecraft reinforcement learning environment with zero HTTP overhead.

---

## 🌟 Features

- ⚡ **Pure Ray Architecture** - Direct RPC communication, no HTTP overhead
- 🚀 **Parallel Creation** - 10 environments in ~2 min vs 20 min serial
- 📁 **Auto Path Isolation** - Each env gets `output/env-{uuid}/`
- 🎮 **Gym-compliant** - Standard `reset()` and `step()` interface
- 🌐 **Distributed Ready** - Scale across multiple machines
- 🎬 **Built-in Recording** - Automatic MP4 recording per episode

---

## 🚀 Quick Start

```python
from raycraft import MCRayClient

# Batch create 10 environments (parallel, ~2 min)
uuids = MCRayClient.create_batch(
    num_envs=10,
    config_path="configs/kill/kill_zombie_with_record.yaml"
)

# Connect to a specific environment
client = MCRayClient(uuid=uuids[0])

# Standard Gym interface
obs = client.reset()
result = client.step('[{"action": "forward"}]')
print(f"Reward: {result.reward}, Done: {result.done}")

# Recording saved to: output/env-{uuid[:8]}/episode_0.mp4
client.close()
```

---

## 📦 Installation

```bash
# Clone with MineStudio submodule
git clone --recursive https://github.com/YOUR_ORG/raycraft
cd raycraft

# Install dependencies
pip install -e .

# Or manually init submodule
git submodule update --init --recursive
```

---

## 🏗️ Architecture

```
User Code
  └─→ MCRayClient.create_batch(10)
       ├─→ MCRayClient(uuid1) ──┐
       ├─→ MCRayClient(uuid2) ──┤
       └─→ MCRayClient(uuid3) ──┴─→ EnvPool → MCEnvActor → MCSimulator → MinecraftSim
                                     (Global)   (Ray Actor)  (Env Logic)    (Minecraft)
```

**Key Components:**
- **MCRayClient** - User-facing interface with Gym API
- **EnvPool** - Global UUID-based environment registry
- **MCEnvActor** - Ray actor wrapping each environment instance
- **MCSimulator** - Environment logic adapter
- **MinecraftSim** - MineStudio's Minecraft simulation

---

## ⚙️ Configuration

Create YAML configs in `configs/`:

```yaml
# configs/my_task.yaml
defaults:
  - base
  - _self_

env:
  seed: 42
  preferred_spawn_biome: "forest"
  timestep_limit: 2000

text: "Kill 10 zombies"

custom_init_commands:
  - "/time set day"
  - "/give @p minecraft:diamond_sword 1"
```

---

## 📚 Examples

See `examples/` directory:
- `mvp1_basic_test.py` - Basic usage and batch creation
- `test_batch_create.py` - Advanced batch operations
- `mvp1_migration_guide.py` - Migrating from other frameworks

---

## 🎯 Use Cases

- **Reinforcement Learning** - Train agents in Minecraft
- **Curriculum Learning** - Gradually increase task difficulty
- **Multi-task Learning** - Train on diverse tasks in parallel
- **Distributed Training** - Scale across GPU clusters

---

## 📖 Documentation

- [Architecture](docs/architecture.md) - Design principles and structure
- [Ray Design](docs/ray-design.md) - Pure Ray implementation details
- [API Reference](docs/api_reference.md) - Complete API documentation

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📄 License

MIT License - see LICENSE file

---

## 🙏 Powered By

- [MineStudio](https://github.com/CraftJarvis/MineStudio) - Minecraft simulation engine
- [Ray](https://github.com/ray-project/ray) - Distributed computing framework
- [OpenAI Gym](https://github.com/openai/gym) - RL environment standard

---

## 📧 Contact

Issues: https://github.com/YOUR_ORG/raycraft/issues
