# SCI-JEPA

Two training stacks live here. [`SCI-JEPA-training/`](SCI-JEPA-training/README.md) is Paper A: the `sci-jepa` command and the dashboard. [`Analysis/`](Analysis/README.md) is Graph-JEPA pretraining. It uses Lightning and writes into that same dashboard. This file is how to install and run the Graph-JEPA side.

| Read | For |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | What each folder does and how the parts connect |
| [Analysis/train/README.md](Analysis/train/README.md) | Which `train/` script is which (several names are misleading) |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Branches, pull requests, uploading through the web UI |
| [AGENTS.md](AGENTS.md) | Instructions that Claude Code, Codex and Cursor read |
| [docs/sciku-bench-paper-guide.md](docs/sciku-bench-paper-guide.md) | The SciKU-Bench reasoning-benchmark paper plan |

## Install

The project uses [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --group dev
uv sync --extra data --group dev   # sentence-transformers, only to embed raw papers
```

`uv sync` creates `.venv` and installs PyTorch, PyTorch Geometric, and Lightning. The model also imports `torch-scatter`, which is not published as a macOS wheel on PyPI (PyTorch Geometric 2.8 no longer depends on it). After sync, install the build that matches your torch:

```bash
uv run python -c "import torch; print(torch.__version__)"
uv pip install torch-scatter --find-links https://data.pyg.org/whl/torch-2.5.0+cpu.html
```

Change `2.5.0+cpu` to your torch version and CUDA tag (`cu124`, `cu126`, ...).

## Train

From `Analysis/` (or via the helper, which cds there):

```bash
bash Analysis/run_local.sh papers device 0
bash Analysis/run_local.sh zinc
```

The same entry from the uv environment:

```bash
uv run graph-jepa train papers device 0
```

`train.backend` defaults to `lightning`. Set `train.backend legacy` to use the original loops. Papers use the VICReg objective (`train.objective: vicreg`); the TU and ZINC scripts use the distance objective. The target network is updated with `train.ema_scope: full` on every step.

One process can use several GPUs, or several seeds can run at once:

```yaml
train:
  devices: 2          # DDP across 2 GPUs, one seed at a time
  strategy: ddp
  parallel_runs: false
```

```yaml
train:
  devices: 0          # keep the single GPU in `device`
  parallel_runs: true # one seed per GPU
```

`parallel_runs` needs `python -m train.<dataset>` (the helper does this) and at least two visible GPUs. It is ignored when DDP is on.

## Logs and the dashboard

Paper A already has the dashboard in `SCI-JEPA-training` (`sci-jepa serve`, or `bash SCI-JEPA-training/scripts/present_dashboard.sh`). Graph-JEPA training writes into that same folder: `runs.json` plus `runs/<name>/live.json`. The loss curve is the **Training dynamics** chart (`train JEPA loss`). The Graphs tab includes the Graphify map of the repository when `graphify-out/graph.html` exists.

Each seed also keeps a local copy:

```
runs/<dataset-timestamp>/seed-<seed>/
  events.jsonl
  config.yaml
  summary.json
  tb/                 # TensorBoard
```

Open the dashboard:

```bash
uv run graph-jepa dashboard
# http://127.0.0.1:8765
```

Point a run at another dashboard folder with `SCIJEPA_DASHBOARD_ROOT` or `logging.dashboard_root`. The default folder is `SCI-JEPA-training/runs/paper_a_results/dashboard`. The Paper A command stays `sci-jepa`; Graph-JEPA training is `graph-jepa`, so the two entry points do not overwrite each other.

## Data paths

Datasets resolve to `Analysis/dataset/<name>` unless you override them.

| Variable | Meaning |
|---|---|
| `SCIJEPA_DATA_ROOT` | Parent of `papers/`, `ZINC/`, `TUD/`, `EXP/` |
| `cfg.data.root` | Full root for the dataset in this run |
| `SCIJEPA_CHECKPOINT_ROOT` | Reasoning-probe checkpoints |

The SLURM script `Analysis/JEPA-rea.sh` sets the first and the third from `PROJECT_DIR`.

## Code graph

Graphify builds a graph of the repository's functions, classes, files and imports, and groups them into named communities. [ARCHITECTURE.md](ARCHITECTURE.md#generated-views-graphify) lists the views; start with `graphify-out/ARCHITECTURE.html` (one diagram per project area) or `graphify-out/GRAPH_TREE.html` (collapsible tree).

One-time setup on each machine:

```bash
uv tool install "graphifyy[mcp]"
graphify hook install          # refresh graph.json after every commit, checkout and merge
```

Refresh names and views by hand:

```bash
bash scripts/graph-refresh.sh            # incremental update, name new communities, rebuild views
bash scripts/graph-refresh.sh --relabel  # rename every community (uses the claude CLI)
```

Coding agents are wired to query the graph before reading files:

| Agent | Instructions | Graph access |
|---|---|---|
| Claude Code | `CLAUDE.md` → `AGENTS.md` | PreToolUse hook in `.claude/settings.json`, MCP server in `.mcp.json`, `/graphify` skill |
| Codex (and any agent that reads `AGENTS.md`) | `AGENTS.md` | `.codex/hooks.json`, `.codex/skills/graphify` |
| Cursor | `.cursor/rules/graphify.mdc` | MCP server in `.cursor/mcp.json` |

All three MCP configs run `scripts/graphify-mcp.sh`. Restart the editor after changing them.
