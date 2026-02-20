#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate GMemory

cd /local3/ericjiang/AgentMemory-new

# OpenAI API config for GPT-4o-mini
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-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}"

python3 tasks/run.py \
    --task sciworld \
    --reasoning io \
    --mas_memory goal-rl \
    --mas_type goal-gcn \
    --model gpt-4o-mini \
    --max_trials 40

