"""Command line for Graph-JEPA training and the existing SCI-JEPA dashboard."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from core.experiment import dashboard_root, ensure_dashboard
from core.paths import analysis_dir, graphify_dir, project_root


HELP = """
Graph-JEPA (Analysis)

  graph-jepa train <dataset> [opts...]   Lightning training (papers, zinc, mutag, ...)
  graph-jepa dashboard [folder]           Open the existing SCI-JEPA dashboard
  graph-jepa context                      Print the Graphify report path

The Paper A command stays `sci-jepa` (SCI-JEPA-training). This command is
`graph-jepa` so the two do not overwrite each other.

Datasets with a train module: papers, zinc, mutag, proteins, dd, exp,
imdbb, imdbm, redditb, redditm.

Examples
  uv run graph-jepa train papers device 0
  uv run graph-jepa dashboard
  PORT=9000 uv run graph-jepa dashboard
""".strip()


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP)
        return 0
    command = args[0]
    if command == "dashboard":
        return _dashboard(args[1:])
    if command == "context":
        return _context()
    if command == "train":
        if len(args) < 2:
            print(HELP)
            return 2
        return _train(args[1], args[2:])
    print(HELP)
    return 2


def _train(name: str, rest: list[str]) -> int:
    module = name if name.startswith("train.") else f"train.{name}"
    os.chdir(analysis_dir())
    os.execv(sys.executable, [sys.executable, "-m", module, *rest])
    return 0


def _dashboard(rest: list[str]) -> int:
    folder = Path(rest[0]) if rest and not rest[0].startswith("-") else dashboard_root()
    ensure_dashboard(folder)
    script = project_root() / "SCI-JEPA-training" / "scripts" / "present_dashboard.sh"
    if not script.is_file():
        print(f"Dashboard script not found: {script}")
        print(f"Prepared folder: {folder}")
        return 1
    os.execv("/bin/bash", ["bash", str(script), str(folder)])
    return 0


def _context() -> int:
    directory = graphify_dir()
    report = directory / "GRAPH_REPORT.md"
    graph = directory / "graph.html"
    print(f"project  {project_root()}")
    print(f"graph    {graph}")
    print(f"report   {report}")
    if report.is_file():
        text = report.read_text(encoding="utf-8")
        print()
        print(text[:5000])
    else:
        print()
        print("No report yet. From the repo root:")
        print("  graphify extract . --code-only")
        print("  graphify cluster-only . --no-label")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
