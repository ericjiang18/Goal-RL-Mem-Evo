# Datasets for G-Memory++

All benchmark datasets are stored in this directory.

## Directory Structure

```
data/
├── alfworld/                    # ALFWorld benchmark
│   ├── alfworld_tasks_suffix.json  # Task definitions
│   ├── json_2.1.1/ -> ~/.cache/alfworld/json_2.1.1  # Game files (symlink)
│   └── logic/ -> ~/.cache/alfworld/logic            # PDDL logic files (symlink)
├── ScreenSpot-Pro/              # GUI Grounding benchmark
│   ├── images/                  # Screenshot images
│   └── annotations/             # Ground truth annotations
├── simple-evals/                # OpenAI simple-evals (auto-downloaded)
├── fever/                       # FEVER fact verification
│   └── fever_dev.jsonl
└── pddl/                        # PDDL planning benchmark
    └── test.jsonl
```

## Setup

Run the download script to set up all datasets:

```bash
bash scripts/download_datasets.sh
```

## Benchmarks

1. **ALFWorld**: Household task planning in TextWorld environment
   - Requires: `alfworld` package and downloaded data via `alfworld-download`

2. **ScreenSpot-Pro**: GUI element grounding from screenshots
   - Data source: HuggingFace `likaixin/ScreenSpot-Pro`

3. **simple-evals**: Standard NLP benchmarks (MMLU, MATH, GPQA, etc.)
   - Data source: HuggingFace datasets (auto-downloaded)

4. **FEVER**: Fact verification benchmark

5. **PDDL**: Planning domain definition language tasks
