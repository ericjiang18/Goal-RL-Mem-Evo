# Goal-RL Memory for ALFWorld

This repository implements a **Goal-Conditioned Reinforcement Learning (Goal-RL)** memory system for autonomous agents, optimized for the ALFWorld benchmark.

## Key Features

- **Goal-RL Framework**: Learns Q-values Q(s, a, g) to guide agent actions toward goals.
- **Hindsight Experience Replay (HER)**: Learns from failures by relabeling them as successful outcomes for achieved states.
- **Skill Mining**: Automatically extracts reusable skills from successful trajectories.
- **Efficient Architecture**: Streamlined single-executor design with minimal LLM overhead (~1 call per step).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ericjiang18/Goal-RL-Mem-Evo.git
    cd Goal-RL-Mem-Evo
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install ALFWorld**:
    Follow instructions at [alfworld](https://github.com/alfworld/alfworld) to install and download data:
    ```bash
    pip install alfworld
    export ALFWORLD_DATA=~/.cache/alfworld  # Or your data path
    alfworld-download
    ```

## Configuration

Set up your API keys in `.env` (or environment variables):

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://api.openai.com/v1"
```

## Running ALFWorld

Run the optimized Goal-RL agent on ALFWorld tasks:

```bash
# Run with GPT-4o-mini (recommended for efficiency)
bash scripts/run_alfworld_gpt4omini.sh
```

### Script Details

The script executes:

```bash
python3 tasks/run.py \
    --task alfworld \
    --reasoning io \
    --mas_memory goal-rl \
    --mas_type goal-gcn \
    --model gpt-4o-mini \
    --max_trials 30
```

- `--mas_memory goal-rl`: Enables the Goal-RL + Skill Mining memory architecture.
- `--mas_type goal-gcn`: Uses the simplified single-executor agent workflow.

## Results

Results (logs, Q-tables, skills) are saved in `.db/gpt-4o-mini/alfworld/goal-gcn/goal-rl/`.
