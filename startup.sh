# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download dataset
DATASET=shakespeare NUM_NODES=2 uv run precache_dataset.py

# 4. Manually run a single training experiment (~30 min)
uv run evaluate.py