"""Infrastructure that does not need a GPU or a dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path

from yacs.config import CfgNode as CN

from core.config import resolve_config, set_cfg
from core.experiment import Experiment
from core.paths import data_root, papers_layout
from core.seeds import seeds_for


class _Cfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_defaults_select_lightning():
    cfg = set_cfg(CN())
    assert cfg.train.backend == "lightning"
    assert cfg.train.objective == "auto"
    assert cfg.train.ema_scope == "full"
    assert cfg.jepa.ema_momentum_end == 0.999
    assert cfg.logging.root == "runs"


def test_resolve_config_finds_papers_yaml():
    path = Path(resolve_config("papers.yaml"))
    assert path.name == "papers.yaml"
    assert path.is_file()


def test_data_root_is_inside_the_repo(monkeypatch):
    monkeypatch.delenv("SCIJEPA_DATA_ROOT", raising=False)
    root = Path(data_root(None, "papers"))
    assert root.name == "papers"
    assert root.parent.name == "dataset"
    layout = papers_layout()
    assert layout["raw"].endswith(f"papers{os.sep}raw")
    assert layout["hetero"].endswith("hetero_graphA.pt")


def test_data_root_honors_explicit_override():
    cfg = _Cfg(data=_Cfg(root="/tmp/sci-jepa-papers"))
    assert data_root(cfg, "papers") == "/tmp/sci-jepa-papers"


def test_seeds_match_the_historical_lists():
    cfg = _Cfg(seed=None, train=_Cfg(runs=2))
    assert seeds_for(cfg, "split") == [21, 42]
    cfg = _Cfg(seed=7, train=_Cfg(runs=4))
    assert seeds_for(cfg, "kfold") == [7]
    assert cfg.train.runs == 1


def test_experiment_writes_existing_dashboard(tmp_path):
    dash = tmp_path / "dashboard"
    experiment = Experiment(
        tmp_path / "runs" / "toy-group" / "seed-3",
        {"group": "toy-group", "run_id": "seed-3", "seed": 3, "dataset": "toy"},
        dashboard=dash,
    )
    experiment.event("epoch", epoch=0, metrics={"train_loss": 1.25, "val_loss": 1.1})
    experiment.write_summary({"seed": 3, "train_loss": 1.25, "mae": 0.4})

    catalog = json.loads((dash / "runs.json").read_text(encoding="utf-8"))
    assert catalog[0]["name"] == "toy__toy-group__seed3"
    assert catalog[0]["path"] == "runs/toy__toy-group__seed3/live.json"
    live = json.loads((dash / "runs" / "toy__toy-group__seed3" / "live.json").read_text())
    assert live["status"] == "done"
    assert live["history"][0]["jepa"] == 1.25
    assert live["history"][0]["val_loss"] == 1.1
    assert live["epoch_results"] == []
    assert live["final"]["mae"] == 0.4
    assert (dash / "index.html").is_file()


def test_objective_math():
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")
    from core.objectives import interpolate_momentum, regression_loss, resolve_plan

    assert interpolate_momentum(0, 10, 0.99, 0.999) == 0.99
    assert abs(interpolate_momentum(10, 10, 0.99, 0.999) - 0.999) < 1e-9
    zeros = torch.zeros(2, 3, 4)
    assert float(regression_loss(zeros, zeros, 0, 0.5)) == 0.0

    cfg = set_cfg(CN())
    cfg.dataset = "papers"

    def train():
        return None

    train.__module__ = "train.papers"
    assert resolve_plan(cfg, train) == ("vicreg", "full")
    cfg.dataset = "ZINC"
    train.__module__ = "train.zinc"
    assert resolve_plan(cfg, train)[0] == "distance"
