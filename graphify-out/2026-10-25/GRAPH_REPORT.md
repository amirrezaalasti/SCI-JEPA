# Graph Report - SCI-JEPA  (2026-10-24)

## Corpus Check
- 162 files · ~191,388 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 9 file(s) not represented in the graph (top: (none) 4, .sbatch 4, .env 1)

## Summary
- 2460 nodes · 5761 edges · 138 communities (95 shown, 43 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 309 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `fc71f8fe`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Config and Data Setup
- experiment.py
- Heterogeneous Graph Builder
- Paper Reasoning JEPA
- Link Prediction Model
- Retrieval Stage Pipeline
- Graph Positional Encoding
- Reasoning Experiments
- Information Theory Metrics
- Paper Data Parsing
- Rebuttal Diagnostic Statistics
- worldview.py
- Analysis Figures
- numpy
- PDF Report Generation
- Pooling and Hyperbolic Encoding
- JEPA Encoder Head
- findings.py
- Hyperbolic Space Reasoning
- Retrieval Experiment Analysis
- GJEPA Reasoning Evaluation
- Reference Model Entropy
- objectives.py
- Experiment Shell Scripts
- engine.py
- Feature Encoders
- Training Loop Execution
- _methods
- JEPA Graph Model
- MLP
- MLP Mixer Attention
- LLM Rebuttal Scripts
- Contrastive Learning Stages
- Corpus Data Loading
- Graph Transformer Encoder
- core/config.py
- DenseSystem
- ku_graph.py
- Paper A — Constraint-Grounded Latent Prediction (Track A code)
- Training Callbacks
- Multihead Attention Module
- Multihead Attention Module
- Multihead Attention Module
- JEPA Environment Config
- LLM Rebuttal Script
- to_device
- Evaluation and Encoding
- reporting/figures.py
- JEPA Model Core
- build_blob
- Graphormer Encoder Layer
- Hadamard Encoder Layer
- Dataset Split Logic
- 6.3 `data/ku_graph.py`
- Cluster Job Scripts
- Parquet Normalization
- Diagnostics CLI Driver
- graph_viz.py
- 8.2 `run_experiment(cfg, gi, exp, arms, log, live_root, resume, shard, name)`
- Analysis Local Runner
- JSON Schema Text Extraction
- GATConv
- test_split.py
- baselines.py
- MCP Tool Config
- Graphify MCP Script
- SCI-JEPA Package
- Evaluation Pipeline
- Sampler
- Inference and Embedding
- report.py
- Training Instance Builder
- Validity and Faithfulness
- Training Telemetry
- Evaluation CLI
- Diagnostics Evaluation
- Parquet Test Suite
- Output Schema Tests
- serve.py
- SciKU Bench Paper Guide
- LLM Evaluation System
- PaperAModel
- Training Test Suite
- Neuron Interpretation
- SAM Optimizer Module
- sys
- GCNConv
- bench_compare.py
- Record Normalization Tests
- Knowledge Unit Schema
- Parquet Writer
- Hardware Monitor
- Project README
- os
- Soft Constraint Loss
- System Hardware Info
- get_index
- Evaluation Protocol
- test_package.py
- Identity
- Knowledge Graph Index
- Project Documentation Files
- gnn_wrapper.py
- Project Setup Guide
- SetTransformerPool
- SCI-JEPA: technical code documentation
- bench_kit.sh
- GraphDataModule
- HyperbolicAnchors
- ResumableEmbedder
- Parquet Output Tests
- GatedGraphConv
- Cluster Operations Guide
- Local Run Environment Config
- Planar SAT Dataset
- Paper Data Pipeline
- make_blocks
- build_minimal_pairs
- Set Pooling Layer
- SCI-JEPA Overview Docs
- Training Scaling Report
- iter_stored
- Training Infrastructure Docs
- JEPA Predictor Module
- Project Documentation
- Summary Download Script
- Dashboard Presentation Script
- Graph-JEPA Package
- Contributing Guidelines
- Architecture Docs
- MCP Graphify Config
- Graph Refresh Script
- Claude Instructions

## God Nodes (most connected - your core abstractions)
1. `Sampler` - 39 edges
2. `train_one()` - 38 edges
3. `PaperAConfig` - 37 edges
4. `to_device()` - 33 edges
5. `build_hetero_graph()` - 32 edges
6. `main()` - 31 edges
7. `evaluate()` - 28 edges
8. `main()` - 26 edges
9. `device()` - 26 edges
10. `update_cfg()` - 25 edges

## Surprising Connections (you probably didn't know these)
- `Graph-JEPA (`Analysis/`)` --references--> `GraphJepa`  [INFERRED]
  ARCHITECTURE.md → Analysis/core/model.py
- `Graph-JEPA (`Analysis/`)` --references--> `run_k_fold()`  [INFERRED]
  ARCHITECTURE.md → Analysis/core/trainer.py
- `Configuration` --references--> `PaperAConfig`  [INFERRED]
  SCI-JEPA-training/README.md → SCI-JEPA-training/src/sci_jepa/config.py
- `How the split is made (`src/sci_jepa/data/split.py`)` --references--> `domain_of()`  [INFERRED]
  SCI-JEPA-training/docs/data.md → SCI-JEPA-training/src/sci_jepa/data/ku_schema.py
- `Benchmark validity (validity.py)` --references--> `agreement()`  [INFERRED]
  SCI-JEPA-training/docs/paper_a.md → SCI-JEPA-training/src/sci_jepa/evaluation/validity.py

## Import Cycles
- None detected.

## Communities (138 total, 43 thin omitted)

### Community 0 - "Config and Data Setup"
Cohesion: 0.10
Nodes (25): build_parser(), merge_opts(), option_docs(), print_options(), resolve_config(), _shortcut_value(), _split_opts(), update_cfg() (+17 more)

### Community 1 - "experiment.py"
Cohesion: 0.12
Nodes (21): _context(), _dashboard(), main(), _tensorboard(), _train(), _atomic(), dashboard_root(), ensure_dashboard() (+13 more)

### Community 2 - "Heterogeneous Graph Builder"
Cohesion: 0.16
Nodes (6): analyze_schema(), build_homogeneous_reasoning_graph(), build_ownership(), compute_patch_rwse(), connectivity_report(), main()

### Community 3 - "Paper Reasoning JEPA"
Cohesion: 0.09
Nodes (23): _apply_masked_edges_to(), build_incident_map(), build_paper_target_map(), collapse_stats(), get_partition(), GNNEncoder, JEPA, main() (+15 more)

### Community 4 - "Link Prediction Model"
Cohesion: 0.07
Nodes (15): _agg_aspect(), _agg_flat(), EdgeDecoder, GNNEncoder, link_metrics(), LinkModel, load_claim_reasoner(), main() (+7 more)

### Community 5 - "Retrieval Stage Pipeline"
Cohesion: 0.11
Nodes (20): _align_to_papers(), _bank_ranks(), _cross_cell(), feature_bank(), _geometry(), _get_x(), hetero_features(), main() (+12 more)

### Community 6 - "Graph Positional Encoding"
Cohesion: 0.11
Nodes (13): LapPE(), random_walk(), RWSE(), cal_coarsen_adj(), combine_subgraphs(), combine_subgraphs_jepa(), GraphJEPAPartitionTransform, PositionalEncodingTransform (+5 more)

### Community 7 - "Reasoning Experiments"
Cohesion: 0.12
Nodes (24): bits_carried(), bootstrap_mrr_ci(), build_cost(), build_membership(), build_query_set(), chance_mrr(), _cos_loss(), _fmt_hms() (+16 more)

### Community 8 - "Information Theory Metrics"
Cohesion: 0.12
Nodes (23): eff_rank_only(), compute_tap(), dc_stats(), estimate_rho_real(), fit_bank_transform(), information_ladder(), measured_redundancy(), node_type_degrees() (+15 more)

### Community 9 - "Paper Data Parsing"
Cohesion: 0.09
Nodes (11): _aspect_text(), build_hetero_graph(), _claim_records(), _coarse(), _coerce_summary(), _get_claims_list(), _iter_raw_records(), _paper_id() (+3 more)

### Community 10 - "Rebuttal Diagnostic Statistics"
Cohesion: 0.10
Nodes (29): bits_recovered(), bootstrap_ci(), device(), mrr(), permutation_p(), ranks_from_banks(), recall_at(), spectral_decay_kappa() (+21 more)

### Community 11 - "worldview.py"
Cohesion: 0.19
Nodes (12): ablate_structure(), add_action(), collate(), action_eval(), _capture(), examples(), _explain(), _predict() (+4 more)

### Community 12 - "Analysis Figures"
Cohesion: 0.17
Nodes (25): _chance(), fig_centering(), fig_dc_ratio(), fig_embedding_geometry(), fig_encoder_depth(), fig_fix_bars(), fig_information_ladder(), fig_ladder() (+17 more)

### Community 13 - "numpy"
Cohesion: 0.07
Nodes (29): hyperbolic_dist(), variance_loss(), variance_loss(), bm25_ranks(), hard_pool(), pooled_ranks(), protocol_r_frame(), training_free_oracle() (+21 more)

### Community 14 - "PDF Report Generation"
Cohesion: 0.17
Nodes (9): method_family(), llm_pages(), _llm_rows(), build(), _display_name(), _f(), _g(), load() (+1 more)

### Community 15 - "Pooling and Hyperbolic Encoding"
Cohesion: 0.10
Nodes (7): AttnPool, GINEncoder, mlp3(), MultiStatPool, GraphJEPAMulti, relhetero_pool(), StructBindPool

### Community 16 - "JEPA Encoder Head"
Cohesion: 0.11
Nodes (8): _ema_(), _etkey(), _gpu_peak_gb(), HeteroEncoder, JEPAHead, _sample_rows(), Timer, train_encoder_variant()

### Community 17 - "findings.py"
Cohesion: 0.13
Nodes (18): 12.2 Result JSON (one per arm × seed), analyse(), _arm(), _avg(), _boot_task(), bootstrap(), _ci(), compare() (+10 more)

### Community 18 - "Hyperbolic Space Reasoning"
Cohesion: 0.20
Nodes (17): collapse_stats(), eval_patchret(), eval_probe(), lorentz_expmap0(), lorentz_inner(), lorentz_sqdist(), mean_std(), project_target() (+9 more)

### Community 19 - "Retrieval Experiment Analysis"
Cohesion: 0.20
Nodes (9): aspect_feature_matrix(), bm25_retrieval(), _fmt(), _pick_device(), sbert_embed(), stage_latent(), stage_oracle(), strip_shared_ngrams() (+1 more)

### Community 20 - "GJEPA Reasoning Evaluation"
Cohesion: 0.11
Nodes (19): build_cost_summary(), eval_patchret_cosine(), _fmt_hms(), _gpu_peak_gb(), _gpu_reset_peak(), main(), aspect_presence(), build_patch_index() (+11 more)

### Community 21 - "Reference Model Entropy"
Cohesion: 0.13
Nodes (8): effective_rank(), set_seed(), conditional_variance_floor(), _grad_norms(), _mlp(), RefJEPA, train_reference_jepa(), _rank_probe()

### Community 22 - "objectives.py"
Cohesion: 0.19
Nodes (14): _batch_size(), covariance_loss(), distance_step(), prediction_loss(), prepare_batch(), regression_loss(), resolve_plan(), vicreg_step() (+6 more)

### Community 23 - "Experiment Shell Scripts"
Cohesion: 0.16
Nodes (16): arm_err(), CUDA_VISIBLE_DEVICES, HF_HOME, MPLBACKEND, MPLCONFIGDIR, mute_err(), p4_build_extra(), p4_has() (+8 more)

### Community 24 - "engine.py"
Cohesion: 0.07
Nodes (37): _barrier(), _check_resume(), _checkpoint_dir(), _devices(), _fit_kfold(), fit_one_seed(), fit_one_seed_cfg(), _fit_split() (+29 more)

### Community 25 - "Feature Encoders"
Cohesion: 0.29
Nodes (6): covariance_loss(), AtomEncoder(), BondEncoder(), DiscreteEncoder(), FeatureEncoder(), LinearEncoder()

### Community 26 - "Training Loop Execution"
Cohesion: 0.05
Nodes (34): 8.1 `train_one(cfg, gi, seed, log, live, save_dir, ckpt_path)`, step(), describe(), minimal_pair(), reasoning_index(), write_progress(), training_step(), _autocast() (+26 more)

### Community 27 - "_methods"
Cohesion: 0.12
Nodes (8): BM25Index, _fingerprint(), get_bm25_index(), strip_ngram_overlap(), _tok(), _lex_idx(), _methods(), _query_idx()

### Community 29 - "MLP"
Cohesion: 0.16
Nodes (3): MLP, VNUpdate, GINEConv

### Community 30 - "MLP Mixer Attention"
Cohesion: 0.18
Nodes (4): MLPMixer, FeedForward, MixerBlock, MLPMixer

### Community 31 - "LLM Rebuttal Scripts"
Cohesion: 0.13
Nodes (14): GJEPA_ARXIV_PROC, GJEPA_ARXIV_RAW, GJEPA_EXTRACTOR_A, GJEPA_EXTRACTOR_B, GJEPA_ROOT, HF_HOME, MPLBACKEND, MPLCONFIGDIR (+6 more)

### Community 32 - "Contrastive Learning Stages"
Cohesion: 0.20
Nodes (13): _infonce_fullpool(), _infonce_hard(), _mm(), pca2_of(), protocol_r_model(), spectrum_of(), stage_centering(), stage_encoder() (+5 more)

### Community 33 - "Corpus Data Loading"
Cohesion: 0.11
Nodes (12): aspect_features(), CorpusSpec, _dig(), _flat(), get_spec(), _iter_records(), load_graph(), load_texts() (+4 more)

### Community 34 - "Graph Transformer Encoder"
Cohesion: 0.22
Nodes (4): GTEncoderLayer, Graph, Kernel, Standard

### Community 35 - "core/config.py"
Cohesion: 0.14
Nodes (15): set_cfg(), data_root(), papers_layout(), _Cfg, test_data_root_honors_explicit_override(), test_data_root_is_inside_the_repo(), test_defaults_select_lightning(), test_every_trainer_key_has_help() (+7 more)

### Community 36 - "DenseSystem"
Cohesion: 0.09
Nodes (13): 9.3 Comparison systems: `evaluation/baselines.py`, BM25System, DenseSystem, _encode_chunked(), encoder_system(), gi_mean(), make_system(), _split_ids() (+5 more)

### Community 37 - "ku_graph.py"
Cohesion: 0.19
Nodes (7): data_fingerprint(), embed_cache_dir(), iter_records(), iter_selected(), shard_files(), load_manifest(), split_files()

### Community 38 - "Paper A — Constraint-Grounded Latent Prediction (Track A code)"
Cohesion: 0.08
Nodes (23): 5.1 Main groups of `PaperAConfig`, Benchmark validity (validity.py), Data split, Diagnostics, always reported together, Knowledge units (KUs), KU modes, Leak-safe masking, Live dashboard and graph views (+15 more)

### Community 43 - "JEPA Environment Config"
Cohesion: 0.20
Nodes (9): CUDA_VISIBLE_DEVICES, HF_HOME, PYTHONUNBUFFERED, SCIJEPA_CHECKPOINT_ROOT, SCIJEPA_DASHBOARD_ROOT, SCIJEPA_DATA_ROOT, SENTENCE_TRANSFORMERS_HOME, JEPA-rea.sh script (+1 more)

### Community 44 - "LLM Rebuttal Script"
Cohesion: 0.20
Nodes (9): HF_HOME, MPLBACKEND, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONPATH, PYTHONUNBUFFERED, SENTENCE_TRANSFORMERS_HOME, run_LLM_rebuttal_w126.sh script (+1 more)

### Community 45 - "to_device"
Cohesion: 0.12
Nodes (9): gpu_step_time(), step(), timed_nb(), to_device(), fit_lightning(), __init__(), transfer_batch_to_device(), _rank_plan() (+1 more)

### Community 46 - "Evaluation and Encoding"
Cohesion: 0.13
Nodes (14): 16. Extending the code, 9.2 Reasoning tasks: `evaluation/tasks.py`, hard_pool(), add_domains(), add_periods(), _by_kind(), _consistency(), _dom() (+6 more)

### Community 47 - "reporting/figures.py"
Cohesion: 0.23
Nodes (13): timeline(), llm_figure(), export(), cell(), _fig_legend(), _footnote(), _hbar(), _label() (+5 more)

### Community 49 - "build_blob"
Cohesion: 0.18
Nodes (7): Stand-ins for data that does not exist yet, build_proxy_edges(), build_blob(), keep_record(), make_splits(), select_balanced(), _temperature_quota()

### Community 52 - "Dataset Split Logic"
Cohesion: 0.18
Nodes (11): _as_text(), ku_fix_drops(), key(), main(), make_split(), parquet_files(), parse_args(), read_rows() (+3 more)

### Community 53 - "6.3 `data/ku_graph.py`"
Cohesion: 0.18
Nodes (5): 6.3 `data/ku_graph.py`, build_context_texts(), HashEmbedder, make_embedder(), STEmbedder

### Community 54 - "Cluster Job Scripts"
Cohesion: 0.08
Nodes (25): DATASET, MKL_NUM_THREADS, OMP_NUM_THREADS, PAPER_A_PARQUET, PYTHONUNBUFFERED, run_paper_a.sh script, TOKENIZERS_PARALLELISM, tunnel() (+17 more)

### Community 55 - "Parquet Normalization"
Cohesion: 0.11
Nodes (13): build_unified_schema(), _canon(), _check_paths(), _env(), free_arrow_memory(), load_progress(), peak_mb(), _progress_path() (+5 more)

### Community 56 - "Diagnostics CLI Driver"
Cohesion: 0.09
Nodes (17): _banner(), _corpus_has_json(), _has_ok(), main(), _parse_ints(), _parse_stages(), _result_dict(), StageTimeout (+9 more)

### Community 57 - "graph_viz.py"
Cohesion: 0.29
Nodes (10): corpus_html(), _edge(), _hub_node(), instance_html(), _ku_node(), paper_html(), short_id(), _snip() (+2 more)

### Community 58 - "8.2 `run_experiment(cfg, gi, exp, arms, log, live_root, resume, shard, name)`"
Cohesion: 0.11
Nodes (10): 14. Performance engineering, 8.2 `run_experiment(cfg, gi, exp, arms, log, live_root, resume, shard, name)`, 8. Training, _RelBias, _claim(), expand_llm_arms(), llm_short_name(), missing_runs() (+2 more)

### Community 59 - "Analysis Local Runner"
Cohesion: 0.40
Nodes (4): PATH, PYTHONUNBUFFERED, SCIJEPA_DATA_ROOT, run_local.sh script

### Community 60 - "JSON Schema Text Extraction"
Cohesion: 0.40
Nodes (5): dump_json_schema(), walk(), extract_text(), load_texts(), _walk_collect()

### Community 62 - "test_split.py"
Cohesion: 0.31
Nodes (6): balanced_quota(), domain_weights(), test_quota_follows_shares_and_redistributes(), test_quota_zero_means_everything(), test_same_shares_in_every_split(), test_uniform_and_natural_weights()

### Community 63 - "baselines.py"
Cohesion: 0.14
Nodes (14): _agg(), _by_kind(), _consistency_probe(), feats(), _consistency_unsup(), _intervention(), _needed_ids(), _pairs() (+6 more)

### Community 71 - "Evaluation Pipeline"
Cohesion: 0.09
Nodes (16): task_summary(), board(), _bool(), display(), domain_balance_table(), _env(), epoch_table(), epoch_types() (+8 more)

### Community 74 - "report.py"
Cohesion: 0.14
Nodes (14): compute(), figures(), hardware(), infra_pages(), load_systems(), load_timeseries(), _mig_fraction(), monitoring() (+6 more)

### Community 75 - "Training Instance Builder"
Cohesion: 0.07
Nodes (16): shapes(), _add_text_token(), build_instance(), build_view(), collate_views(), _drop_tokens(), _extra_anchors(), _hard_negative() (+8 more)

### Community 76 - "Validity and Faithfulness"
Cohesion: 0.14
Nodes (15): 9.4 Other evaluation modules, synthetic_records(), agreement(), apply_filter(), audit(), _cell(), contamination(), export() (+7 more)

### Community 77 - "Training Telemetry"
Cohesion: 0.14
Nodes (3): _atomic_dump(), _clean(), LiveRun

### Community 78 - "Evaluation CLI"
Cohesion: 0.15
Nodes (14): resolve_device(), cmd_model(), cmd_results(), cmd_system(), fmt(), g(), get_gi(), headline() (+6 more)

### Community 79 - "Diagnostics Evaluation"
Cohesion: 0.11
Nodes (17): 9.1 Main evaluation: `evaluation/diagnostics.py`, 9. Evaluation, perturb_relation(), _boot_diff(), _breakdowns(), centred_cos(), _chance(), chi_stats() (+9 more)

### Community 80 - "Parquet Test Suite"
Cohesion: 0.13
Nodes (4): json_files(), parquet_files(), _record_files(), TestEndToEnd

### Community 82 - "serve.py"
Cohesion: 0.14
Nodes (5): page(), step(), _Quiet, register_graph(), start_server()

### Community 83 - "SciKU Bench Paper Guide"
Cohesion: 0.10
Nodes (19): 10The A\* paper, 11What reviewers will ask, 12Work plan, 1Short answers, 2The five reasoning tasks, 3The key finding: in their current form, three tasks are solved without reasoning, 4Making the tasks require reasoning, 5Why no-training baselines are required (+11 more)

### Community 85 - "PaperAModel"
Cohesion: 0.13
Nodes (5): GraphEncoder, InverseRelationHead, PaperAModel, rwse(), check_invariants()

### Community 86 - "Training Test Suite"
Cohesion: 0.21
Nodes (9): _jepa_losses(), Stop, test_batch_workers_independent_of_worker_count_and_resumable(), dies_at_step_8(), test_checkpoint_of_other_settings_is_ignored(), test_dynamic_shards_run_every_arm_once(), test_llm_arms_expand_one_per_model(), test_resume_matches_uninterrupted() (+1 more)

### Community 87 - "Neuron Interpretation"
Cohesion: 0.15
Nodes (8): _corr(), _label(), _layer_verdict(), neuron_report(), omega2(), _top_group(), _type_centre(), Predictor

### Community 88 - "SAM Optimizer Module"
Cohesion: 0.09
Nodes (7): ASAM, SAM, flatten_cfg(), GraphJepaLitModule, ema_bounds(), ema_update(), interpolate_momentum()

### Community 89 - "sys"
Cohesion: 0.11
Nodes (7): config_logger(), _Tee, fetch_assets(), main(), usage(), ensure_vendor(), test_cli_lists_every_command()

### Community 91 - "bench_compare.py"
Cohesion: 0.12
Nodes (6): energy(), log(), nbytes(), _nv(), Probe, on_train_batch_end()

### Community 92 - "Record Normalization Tests"
Cohesion: 0.19
Nodes (3): normalize_record(), _safe_filename(), TestNormalizer

### Community 93 - "Knowledge Unit Schema"
Cohesion: 0.07
Nodes (24): 6.1 `data/ku_schema.py`, 6.2 `data/split.py`, 6.4 `data/graph_index.py`, 6. Data layer, Research-domain imbalance, active_templates(), _case(), clean() (+16 more)

### Community 95 - "Hardware Monitor"
Cohesion: 0.16
Nodes (5): 10. Reporting, dashboard and infrastructure monitoring, _atomic(), Monitor, progress_dir(), start()

### Community 96 - "Project README"
Cohesion: 0.14
Nodes (14): 1. Environment, 2. Data, 3. Run the pipeline, 🙌 Acknowledgements, 💾 Caches, ⚙️ Configuration, ✨ Features, 🚀 Getting Started (+6 more)

### Community 97 - "os"
Cohesion: 0.27
Nodes (5): render_all(), _save(), get(), paired_bars(), savefig()

### Community 98 - "Soft Constraint Loss"
Cohesion: 0.23
Nodes (3): _keep(), ShapeBank, tnorm()

### Community 99 - "System Hardware Info"
Cohesion: 0.18
Nodes (10): _slurm_end(), collect(), _cpu(), _gpus(), _run(), _slurm(), _storage(), _torch() (+2 more)

### Community 100 - "get_index"
Cohesion: 0.24
Nodes (7): setup_dashboard(), get_index(), audit_tightness(), check_validity_and_findings(), main(), main(), run_experiment()

### Community 101 - "Evaluation Protocol"
Cohesion: 0.15
Nodes (13): 1. The data unit: knowledge units, 2. Split by time: no future papers in training, 3. A query: mask one KU, predict it, 4. Ranking against hard candidates, 5. The five reasoning tasks, 6. Comparison systems: same queries, same candidates, 7. Model diagnostics (models only), 8. Seeds, best epoch and hypotheses (+5 more)

### Community 102 - "test_package.py"
Cohesion: 0.07
Nodes (12): _coerce_summary(), _extract_node_texts(), PapersDataset, parse_paper(), all_modules(), test_cli_help(), test_module_imports(), test_paths_come_from_the_environment() (+4 more)

### Community 105 - "Project Documentation Files"
Cohesion: 0.24
Nodes (4): Changed after the migration (intended), Migration from SCI-JEPA-training/Analysis, What changed for users, Where things went

### Community 107 - "Project Setup Guide"
Cohesion: 0.18
Nodes (11): Commands, Configuration, Documentation, Install, Long runs on a cluster (e.g. 60,000 papers), Quick start, Repository layout, Reproducibility (+3 more)

### Community 109 - "SCI-JEPA: technical code documentation"
Cohesion: 0.10
Nodes (19): 11.1 Scripts, 11.2 One job in `dist` mode, 11.3 Dashboard access, 11. Running on a Slurm cluster, 12.1 Layout, 12. Outputs and file formats, 13. Reproducibility and determinism, 15. Testing (+11 more)

### Community 110 - "bench_kit.sh"
Cohesion: 0.29
Nodes (7): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, run(), bench_kit.sh script, TOKENIZERS_PARALLELISM, TRANSFORMERS_OFFLINE

### Community 116 - "Cluster Operations Guide"
Cohesion: 0.25
Nodes (8): Browser, Grete facts that matter, Long experiments: chained 12-hour jobs, Make the data split, Once, Running on a Slurm cluster (GWDG Grete), Runs, Speed settings

### Community 117 - "Local Run Environment Config"
Cohesion: 0.25
Nodes (7): PAPER_A_INDEX_DIR, PAPER_A_OUT, PAPER_A_PARQUET, PYTHONUNBUFFERED, PYTORCH_ENABLE_MPS_FALLBACK, run_local.sh script, TOKENIZERS_PARALLELISM

### Community 119 - "Paper Data Pipeline"
Cohesion: 0.29
Nodes (7): 1. Download, 2. Normalize, 3. Split (train / validation / test), domain-balanced, Data: download, normalize, split, How the split is made (`src/sci_jepa/data/split.py`), Paper ids, Why the balance is needed: arXiv + PubMed per split

### Community 120 - "make_blocks"
Cohesion: 0.08
Nodes (12): 7.1 Attention with relation biases, 7.2 Encoder families (`make_blocks`), 7. Model, Block, _edges(), GINEBlock, GINEMessage, GPSBlock (+4 more)

### Community 123 - "SCI-JEPA Overview Docs"
Cohesion: 0.25
Nodes (8): Changing hyperparameters, Checkpoints and resuming, Code graph, Data paths, Install, Logs and the dashboard, SCI-JEPA, Train

### Community 124 - "Training Scaling Report"
Cohesion: 0.33
Nodes (6): 1. Where the time went (before), 2. What was changed, 3. Measured effect (after), 4. Accuracy check (before vs after), 5. Files, Scaling SCI-JEPA training to 60k papers

### Community 126 - "Training Infrastructure Docs"
Cohesion: 0.40
Nodes (5): How compute is counted, Long runs: checkpoints and chained jobs, Outputs for the paper, Training infrastructure, What is recorded

### Community 128 - "Project Documentation"
Cohesion: 0.27
Nodes (6): Agent guide, Commands, graphify, Rules for this repository, Reasoning experiments on the heterogeneous paper graph, `train/` file map

### Community 136 - "Contributing Guidelines"
Cohesion: 0.29
Nodes (7): Branches, Code graph, Contributing, Pull requests, What not to commit, Working from a terminal, Working from the GitHub web UI

### Community 138 - "Architecture Docs"
Cohesion: 0.33
Nodes (6): Architecture, Generated views (Graphify), Graph-JEPA (`Analysis/`), How the pieces connect, Paper A (`SCI-JEPA-training/`), The papers pipeline (`Analysis/JEPA-rea.sh`)

## Knowledge Gaps
- **71 isolated node(s):** `bash`, `bash`, `PYTHONUNBUFFERED`, `SCIJEPA_DATA_ROOT`, `SCIJEPA_CHECKPOINT_ROOT` (+66 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 952 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **43 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PaperAConfig` connect `Paper A — Constraint-Grounded Latent Prediction (Track A code)` to `get_index`, `ku_graph.py`, `8.2 `run_experiment(cfg, gi, exp, arms, log, live_root, resume, shard, name)``, `Evaluation Pipeline`, `Inference and Embedding`, `Project Setup Guide`, `Validity and Faithfulness`, `SCI-JEPA: technical code documentation`, `Evaluation and Encoding`, `Evaluation CLI`, `Training Instance Builder`, `Dataset Split Logic`, `Training Test Suite`, `Training Loop Execution`, `baselines.py`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **What connects `bash`, `bash`, `PYTHONUNBUFFERED` to the rest of the system?**
  _71 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Config and Data Setup` be split into smaller, more focused modules?**
  _Cohesion score 0.10303030303030303 - nodes in this community are weakly interconnected._
- **Why does `PapersDataset` connect `test_package.py` to `Config and Data Setup`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Should `experiment.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12181616832779624 - nodes in this community are weakly interconnected._
- **Why does `SCI-JEPA: technical code documentation` connect `SCI-JEPA: technical code documentation` to `Project Documentation Files`, `Evaluation and Encoding`, `Diagnostics Evaluation`, `make_blocks`, `8.2 `run_experiment(cfg, gi, exp, arms, log, live_root, resume, shard, name)``, `Knowledge Unit Schema`, `Hardware Monitor`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Should `Paper Reasoning JEPA` be split into smaller, more focused modules?**
  _Cohesion score 0.08826945412311266 - nodes in this community are weakly interconnected._