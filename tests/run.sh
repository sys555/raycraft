#!/bin/bash
# Quick test runner for dual action format

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

# Check server
curl -sf http://localhost:8000/health > /dev/null || {
    echo "Server not running. Start with: python -m raycraft.http_server"
    exit 1
}

# Run tests
pytest tests/test_dual_action_format.py -v "$@"
