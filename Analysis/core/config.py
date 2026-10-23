import os
import argparse
from pathlib import Path
from yacs.config import CfgNode as CN


def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Dataset name
    cfg.dataset = 'ZINC'
    # Additional num of worker for data loading
    cfg.num_workers = 8
    # Cuda device number, used for machine with multiple gpus
    cfg.device = 0
    # Whether fix the running seed to remove randomness
    cfg.seed = None
    # Custom log file name
    cfg.logfile = None
    # tree depth for TreeDataset
    cfg.depth = -1
    # Number of cross-validation folds
    cfg.k = 2

    # ------------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------------ #
    cfg.train = CN()
    # Total graph mini-batch size
    cfg.train.batch_size = 128
    # Maximal number of epochs
    cfg.train.epochs = 1000
    # Number of runs with random init
    cfg.train.runs = 4
    # Base learning rate
    cfg.train.lr = 0.001
    # number of steps before reduce learning rate
    cfg.train.lr_patience = 20
    # learning rate decay factor
    cfg.train.lr_decay = 0.5
    # L2 regularization, weight decay
    cfg.train.wd = 0.
    # Dropout rate
    cfg.train.dropout = 0.
    # Dropout rate for MLPMixer
    cfg.train.mlpmixer_dropout = 0.
    # A lower bound on the learning rate.
    cfg.train.min_lr = 1e-5
    # optimizer: Adam, SGD, or ASAM (ASAM is a sharpness-aware step on top of SGD)
    cfg.train.optimizer = 'Adam'
    # lightning (default) or legacy (the original hand-written loops)
    cfg.train.backend = 'lightning'
    # auto selects vicreg for papers and distance for the TU / ZINC scripts
    cfg.train.objective = 'auto'
    # full updates every EMA pair; encoder updates only the context encoder
    cfg.train.ema_scope = 'full'
    # 0 keeps the single GPU in cfg.device. >1 turns on DDP across that many GPUs.
    cfg.train.devices = 0
    # auto or ddp
    cfg.train.strategy = 'auto'
    # one seed per GPU, sequential seeds when this is false
    cfg.train.parallel_runs = False
    # 0 disables clipping. The papers run sets this to 1.
    cfg.train.grad_clip = 0.0
    cfg.train.precision = '32-true'
    # SmoothL1 beta. 0.5 matches ZINC/DD/IMDB/REDDIT; MUTAG/PROTEINS/EXP use 1.
    cfg.train.smooth_l1_beta = 0.5
    cfg.train.sim_weight = 1.0
    # import path of the training script, filled in for multi-process seeds
    cfg.train.entry = ''
    # Multiscale training
    cfg.train.multiscale = False
    # ── anti-collapse (VICReg) regularizer weights ──
    cfg.train.var_weight = 1.0      # VICReg variance hinge weight
    cfg.train.cov_weight = 0.04     # VICReg covariance weight

    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()
    # GNN type used, see core.model_utils.pyg_gnn_wrapper for all options
    cfg.model.gnn_type = 'GINEConv'  # change to list later
    # GraphMLPMixer or graph-based multihead attention: [MLPMixer, Hadamard, Standard, Graph, Addictive, Kernel]
    cfg.model.gMHA_type = 'MLPMixer'
    # Hidden size of the model
    cfg.model.hidden_size = 128
    # Number of gnn layers
    cfg.model.nlayer_gnn = 4
    # Number of mlp mixer layers
    cfg.model.nlayer_mlpmixer = 4
    # Pooling type for generaating graph/subgraph embedding from node embeddings
    cfg.model.pool = 'mean'
    # Use residual connection
    cfg.model.residual = True
    # Number of output classes for the papers dataset (coarse field labels)
    cfg.model.papers_nout = 0

    # ------------------------------------------------------------------------ #
    # Positional encoding options
    # ------------------------------------------------------------------------ #
    cfg.pos_enc = CN()
    # Random walk structural encoding
    cfg.pos_enc.rw_dim = 0
    # Laplacian eigenvectors positional encoding
    cfg.pos_enc.lap_dim = 0
    # Patch random walk structural encoding
    cfg.pos_enc.patch_rw_dim = 8
    # Patch PE diffusion steps
    cfg.pos_enc.patch_num_diff = -1

    # ------------------------------------------------------------------------ #
    # Metis patch extraction options
    # ------------------------------------------------------------------------ #
    cfg.metis = CN()
    # Enable Metis partition (otherwise use random partition)
    cfg.metis.enable = True
    # Enable data augmentation
    cfg.metis.online = True
    # The number of partitions
    cfg.metis.n_patches = 32
    # Whether to randomly drop a set of edges before each metis partition
    cfg.metis.drop_rate = 0.3
    # expanding patches with k hop neighbourhood
    cfg.metis.num_hops = 1

    # ------------------------------------------------------------------------ #
    # JEPA options
    # ------------------------------------------------------------------------ #
    cfg.jepa = CN()
    # Use Graph-JEPA
    cfg.jepa.enable = True
    # Number of patches to use as context
    cfg.jepa.num_context = 1
    # Number of patches to use as targets
    cfg.jepa.num_targets = 2  # 4
    # Distance function: 0 = 2d Hyper (Ours), 1 = Euclidean, 2 = Hyperbolic (Poincaré)
    cfg.jepa.dist = 0
    # EMA target-network momentum (anti-collapse)
    cfg.jepa.ema_momentum = 0.996
    # Momentum reached at the last optimization step
    cfg.jepa.ema_momentum_end = 0.999

    # ------------------------------------------------------------------------ #
    # Data and logging
    # ------------------------------------------------------------------------ #
    cfg.data = CN()
    # Full dataset root for this run. Empty uses SCIJEPA_DATA_ROOT or Analysis/dataset.
    cfg.data.root = ''

    cfg.logging = CN()
    cfg.logging.root = 'runs'
    cfg.logging.tensorboard = True
    cfg.logging.jsonl = True
    # Folder with the existing dashboard (index.html, runs.json, runs/).
    # Empty uses SCIJEPA_DASHBOARD_ROOT or
    # SCI-JEPA-training/runs/paper_a_results/dashboard.
    cfg.logging.dashboard_root = ''

    return cfg


# Principle means that if an option is defined in a YACS config object,
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining,
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.


def resolve_config(name: str) -> str:
    """Find a YAML config from the repo root or from ``Analysis/``."""
    if not name:
        return name
    raw = Path(name)
    analysis = Path(__file__).resolve().parents[1]
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend([
        Path.cwd() / name,
        analysis / name,
        analysis / "train" / "configs" / raw.name,
    ])
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return name


def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    config_path = resolve_config(args.config) if args.config else ""
    if config_path and os.path.isfile(config_path):
        cfg.merge_from_file(config_path)

    # Update from command line
    cfg.merge_from_list(args.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())