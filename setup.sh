#!/bin/bash
# Raycraft 快速安装脚本

set -e

echo "===================================="
echo "   Raycraft 安装脚本"
echo "===================================="
echo ""

# 检查 Python 版本
echo "检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"

# 初始化 MineStudio submodule
echo ""
echo "初始化 MineStudio submodule..."
git submodule update --init --recursive

# 安装 raycraft
echo ""
echo "安装 raycraft..."
pip install -e .

# 安装 MineStudio 依赖
echo ""
echo "安装 MineStudio..."
cd MineStudio
pip install -e .
cd ..

echo ""
echo "===================================="
echo "   ✅ 安装完成！"
echo "===================================="
echo ""
echo "快速测试:"
echo "  python -c 'from raycraft import MCRayClient; print(\"✅ Import OK\")'"
echo ""
echo "运行示例:"
echo "  python examples/mvp1_basic_test.py"
echo ""
