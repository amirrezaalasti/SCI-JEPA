# Graph Report - SCI-JEPA  (2026-10-27)

## Corpus Check
- 179 files · ~195,532 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 11 file(s) not represented in the graph (top: (none) 5, .sbatch 4, .env 1)

## Summary
- 3266 nodes · 6786 edges · 196 communities (151 shown, 45 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 383 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9a468e5f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Config and Data Setup
- auth.ts
- Heterogeneous Graph Builder
- Paper Reasoning JEPA
- Link Prediction Model
- Retrieval Stage Pipeline
- Graph Positional Encoding
- Reasoning Experiments
- Information Theory Metrics
- Paper Data Parsing
- push_dashboard.py
- glob
- Analysis Figures
- common.py
- Slide 31 — Backup: Graph-JEPA (the other pipeline in the repository)
- Pooling and Hyperbolic Encoding
- train_variant
- findings.py
- Hyperbolic Space Reasoning
- stage_latent
- GJEPA Reasoning Evaluation
- build_deck.js
- monitor.py
- Experiment Shell Scripts
- engine.py
- Feature Encoders
- Training Loop Execution
- time
- JEPA Graph Model
- _batches
- MLP Mixer Attention
- LLM Rebuttal Scripts
- eval_probe
- Corpus Data Loading
- Graph Transformer Encoder
- GraphJEPAMulti
- DenseSystem
- numpy
- test_push_dashboard.py
- Training Callbacks
- Multihead Attention Module
- Multihead Attention Module
- Multihead Attention Module
- JEPA Environment Config
- LLM Rebuttal Script
- How it works / why it is true
- Brief: build the "Paper A training optimization" deck (pptx + pdf)
- Background you need
- JEPA Model Core
- ku_graph.py
- Graphormer Encoder Layer
- Hadamard Encoder Layer
- Dataset Split Logic
- experiment.py
- Cluster Job Scripts
- Parquet Normalization
- Diagnostics CLI Driver
- graph_viz.py
- _RelBias
- Analysis Local Runner
- JSON Schema Text Extraction
- test_split.py
- baselines.py
- MCP Tool Config
- Graphify MCP Script
- SCI-JEPA Package
- Evaluation Pipeline
- Sampler
- Inference and Embedding
- report.py
- instances.py
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
- os
- Neuron Interpretation
- SAM Optimizer Module
- log.py
- How it works / why it matters
- bench_compare.py
- Record Normalization Tests
- Knowledge Unit Schema
- Parquet Writer
- Hardware Monitor
- Project README
- os
- Soft Constraint Loss
- System Hardware Info
- Background you need
- Evaluation Protocol
- test_package.py
- gnn_wrapper.py
- Knowledge Graph Index
- CODE_DOCUMENTATION.md
- ResGatedGraphConv
- Project Setup Guide
- SetTransformerPool
- SCI-JEPA: technical code documentation
- bench_kit.sh
- Background you need
- banks.py
- Slide 7 — Why fewer bytes means less time
- Parquet Output Tests
- Background you need
- Cluster Operations Guide
- Local Run Environment Config
- Planar SAT Dataset
- Paper Data Pipeline
- make_blocks
- Slide 15 — Energy: more power for half as long = −22 to −30% per KU
- reporting/figures.py
- SCI-JEPA Overview Docs
- Training Scaling Report
- Slide 14 — Change 3, data I/O: relation ids fit in one byte
- Slide 16 — Cost: −46% GPU-hours per S run; DDP trades GPU-hours for time
- JEPA Predictor Module
- Project Documentation
- Background you need
- Slide 19 — DDP: averaging per-GPU gradients is exactly the full-batch gradient
- Summary Download Script
- Dashboard Presentation Script
- Graph-JEPA Package
- data_utils/papers.py
- Contributing Guidelines
- Slide 2 — The answer: half the GPU time, same arithmetic, same MRR
- Architecture Docs
- Slide 13 — Memory: the honest cost is +20–37% reserved memory
- Slide 22 — Which GPU? H100 measured, A100 and L40S pending
- Slide 8 — Evidence: the time disappears exactly in the changed kernels
- Slide 11 — Time: 1.6–1.9× faster per GPU; 4 GPUs a further 2.1–3.0×
- Slide 24 — Repository: Graphify turns the code into a queryable map
- MCP Graphify Config
- Graph Refresh Script
- Claude Instructions
- Slide 18 — With vs without Lightning, measured on 1 H100: no difference
- T
- BM25System
- stages_sim.py
- Slide 1 — Paper A trains 1.8× faster at the same accuracy
- 03-setup.md
- Slide 20 — Scaling: 4 independent runs 3.99× per node, one DDP run 2.06×
- Slide 21 — How many H100s? One per run, four only when time matters
- PapersDataset
- Glossary: every technical word in the deck, in plain language
- encoder_system
- probe.py
- GINEMessage
- check_environment.py
- Slide 23 — Repository: one Lightning runtime in each pipeline
- Background you need
- sci_jepa/cli.py
- assemble.mjs
- train
- NVIDIA H100
- Background you need
- Slide 6 — Change 2, backward: build the loop-invariant one-hot once
- RelBiasAttention
- rect
- has
- compilerOptions
- regenerate.sh
- 6. Data layer
- How it works / why it matters
- GatedGraphConv
- Explanations for the deck "Paper A trains 1.8× faster at the same accuracy"
- 11. Running on a Slurm cluster
- Migration from SCI-JEPA-training/Analysis
- InverseRelationHead
- seg
- Background you need
- hosted.js
- Pusher
- HeteroEncoder
- Background you need
- Hosted dashboard (Vercel)
- HashEmbedder
- ResumableEmbedder
- iter_stored
- Predictor
- vercel.json
- build_minimal_pairs
- STEmbedder
- _rho_make_task

## God Nodes (most connected - your core abstractions)
1. `train_one()` - 61 edges
2. `Sampler` - 40 edges
3. `PaperAConfig` - 37 edges
4. `to_device()` - 36 edges
5. `build_hetero_graph()` - 32 edges
6. `main()` - 31 edges
7. `evaluate()` - 28 edges
8. `device()` - 26 edges
9. `main()` - 26 edges
10. `update_cfg()` - 25 edges

## Surprising Connections (you probably didn't know these)
- `AST-only extraction, no LLM` --references--> `run_lightning()`  [INFERRED]
  meeting-2026-10-26/explanations/24-repo-graphify.md → Analysis/core/engine.py
- `1. Audience, goal, constraints` --references--> `train_one()`  [INFERRED]
  meeting-2026-10-26/deck_brief_for_powerpoint.md → SCI-JEPA-training/src/sci_jepa/training/train.py
- `3. Setup (one slide)` --references--> `train_one()`  [INFERRED]
  meeting-2026-10-26/deck_brief_for_powerpoint.md → SCI-JEPA-training/src/sci_jepa/training/train.py
- `C. The model: attention, JEPA, training` --references--> `train_one()`  [INFERRED]
  meeting-2026-10-26/explanations/00-glossary.md → SCI-JEPA-training/src/sci_jepa/training/train.py
- `What is Paper A / SCI-JEPA?` --references--> `train_one()`  [INFERRED]
  meeting-2026-10-26/explanations/01-title.md → SCI-JEPA-training/src/sci_jepa/training/train.py

## Import Cycles
- None detected.

## Communities (196 total, 45 thin omitted)

### Community 0 - "Config and Data Setup"
Cohesion: 0.17
Nodes (17): build_parser(), merge_opts(), option_docs(), print_options(), resolve_config(), _shortcut_value(), _split_opts(), update_cfg() (+9 more)

### Community 1 - "auth.ts"
Cohesion: 0.06
Nodes (46): @types/node, typescript, @upstash/redis, @vercel/functions, GET(), POST(), safeNext(), current() (+38 more)

### Community 2 - "Heterogeneous Graph Builder"
Cohesion: 0.21
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
Cohesion: 0.10
Nodes (15): LapPE(), random_walk(), RWSE(), unithyper_geodesic(), cal_coarsen_adj(), combine_subgraphs(), combine_subgraphs_jepa(), GraphJEPAPartitionTransform (+7 more)

### Community 7 - "Reasoning Experiments"
Cohesion: 0.09
Nodes (18): bits_carried(), build_cost(), build_membership(), build_query_set(), chance_mrr(), _fmt_hms(), load_artifacts(), main() (+10 more)

### Community 8 - "Information Theory Metrics"
Cohesion: 0.11
Nodes (31): eff_rank_only(), aspect_feature_matrix(), bm25_retrieval(), bootstrap_mrr_ci(), compute_tap(), dc_stats(), estimate_rho_real(), fit_bank_transform() (+23 more)

### Community 9 - "Paper Data Parsing"
Cohesion: 0.10
Nodes (10): _aspect_text(), build_hetero_graph(), _claim_records(), _coarse(), _coerce_summary(), _get_claims_list(), _paper_id(), _paper_text() (+2 more)

### Community 10 - "push_dashboard.py"
Cohesion: 0.17
Nodes (9): encode(), log(), main(), one_round(), payload_bytes(), squeue_snapshot(), thin_history(), walk() (+1 more)

### Community 11 - "glob"
Cohesion: 0.16
Nodes (7): _iter_raw_records(), build(), _display_name(), _f(), _g(), load(), _metrics()

### Community 12 - "Analysis Figures"
Cohesion: 0.20
Nodes (25): _chance(), fig_centering(), fig_dc_ratio(), fig_embedding_geometry(), fig_encoder_depth(), fig_fix_bars(), fig_information_ladder(), fig_ladder() (+17 more)

### Community 13 - "common.py"
Cohesion: 0.11
Nodes (9): effective_rank(), set_seed(), timer(), conditional_variance_floor(), _grad_norms(), _mlp(), RefJEPA, train_reference_jepa() (+1 more)

### Community 14 - "Slide 31 — Backup: Graph-JEPA (the other pipeline in the repository)"
Cohesion: 0.04
Nodes (48): Background: the task names, Background you need, Background you need, Background you need (shared by all five slides), How it works / why it matters, How it works / why it matters, How it works / why it matters, How it works / why it matters (+40 more)

### Community 15 - "Pooling and Hyperbolic Encoding"
Cohesion: 0.10
Nodes (6): AttnPool, GINEncoder, mlp3(), MultiStatPool, HyperbolicAnchors, StructBindPool

### Community 16 - "train_variant"
Cohesion: 0.09
Nodes (15): _cos_loss(), _ema_(), _gpu_peak_gb(), _infonce_fullpool(), _infonce_hard(), _infonce_inbatch(), _jepa_loss(), JEPAHead (+7 more)

### Community 17 - "findings.py"
Cohesion: 0.13
Nodes (17): 12.2 Result JSON (one per arm × seed), analyse(), _arm(), _avg(), _boot_task(), bootstrap(), _ci(), compare() (+9 more)

### Community 18 - "Hyperbolic Space Reasoning"
Cohesion: 0.23
Nodes (15): collapse_stats(), eval_patchret(), lorentz_expmap0(), lorentz_inner(), lorentz_sqdist(), project_target(), run_cell(), to_hyperbola() (+7 more)

### Community 19 - "stage_latent"
Cohesion: 0.40
Nodes (3): _fmt(), stage_latent(), variance_decomposition()

### Community 20 - "GJEPA Reasoning Evaluation"
Cohesion: 0.13
Nodes (19): build_cost_summary(), eval_patchret_cosine(), _fmt_hms(), _gpu_peak_gb(), _gpu_reset_peak(), main(), aspect_presence(), build_patch_index() (+11 more)

### Community 21 - "build_deck.js"
Cohesion: 0.05
Nodes (33): ac, { applyTheme }, CE, CF, CF3, CG, chartOpts(), CL (+25 more)

### Community 22 - "monitor.py"
Cohesion: 0.16
Nodes (6): 10. Reporting, dashboard and infrastructure monitoring, _slurm_end(), start(), area_of(), build_sections(), main()

### Community 23 - "Experiment Shell Scripts"
Cohesion: 0.16
Nodes (16): arm_err(), CUDA_VISIBLE_DEVICES, HF_HOME, MPLBACKEND, MPLCONFIGDIR, mute_err(), p4_build_extra(), p4_has() (+8 more)

### Community 24 - "engine.py"
Cohesion: 0.08
Nodes (32): _barrier(), _check_resume(), _checkpoint_dir(), _devices(), _fit_kfold(), fit_one_seed(), fit_one_seed_cfg(), _fit_split() (+24 more)

### Community 25 - "Feature Encoders"
Cohesion: 0.23
Nodes (7): covariance_loss(), AtomEncoder(), BondEncoder(), DiscreteEncoder(), FeatureEncoder(), LinearEncoder(), variance_loss()

### Community 26 - "Training Loop Execution"
Cohesion: 0.05
Nodes (38): Automatic fallback: `supported()`, Where the facts come from (which repo file), 14. Performance engineering, 8.2 `run_experiment(cfg, gi, exp, arms, log, live_root, resume, shard, name)`, 8. Training, What is recorded, resolve_device(), minimal_pair() (+30 more)

### Community 27 - "time"
Cohesion: 0.09
Nodes (26): train_jepa(), bits_recovered(), bootstrap_ci(), mrr(), permutation_p(), ranks_from_banks(), recall_at(), StageResult (+18 more)

### Community 29 - "_batches"
Cohesion: 0.14
Nodes (6): _batches(), make(), work(), _first(), _StepBuilder, _StepPlan

### Community 30 - "MLP Mixer Attention"
Cohesion: 0.18
Nodes (4): MLPMixer, FeedForward, MixerBlock, MLPMixer

### Community 31 - "LLM Rebuttal Scripts"
Cohesion: 0.13
Nodes (14): GJEPA_ARXIV_PROC, GJEPA_ARXIV_RAW, GJEPA_EXTRACTOR_A, GJEPA_EXTRACTOR_B, GJEPA_ROOT, HF_HOME, MPLBACKEND, MPLCONFIGDIR (+6 more)

### Community 32 - "eval_probe"
Cohesion: 0.38
Nodes (9): eval_probe(), _mm(), pca2_of(), spectrum_of(), stage_centering(), stage_encoder(), stage_fix(), stage_negatives() (+1 more)

### Community 33 - "Corpus Data Loading"
Cohesion: 0.14
Nodes (10): aspect_features(), CorpusSpec, _dig(), _flat(), get_spec(), _iter_records(), load_graph(), load_texts() (+2 more)

### Community 34 - "Graph Transformer Encoder"
Cohesion: 0.22
Nodes (4): GTEncoderLayer, Graph, Kernel, Standard

### Community 35 - "GraphJEPAMulti"
Cohesion: 0.18
Nodes (3): GraphJEPAMulti, oracle_retrieval_test(), relhetero_pool()

### Community 36 - "DenseSystem"
Cohesion: 0.18
Nodes (4): DenseSystem, gi_mean(), __init__(), _unit()

### Community 37 - "numpy"
Cohesion: 0.06
Nodes (29): calculate_stats(), hyperbolic_dist(), test(), train(), test(), train(), test(), train() (+21 more)

### Community 38 - "test_push_dashboard.py"
Cohesion: 0.22
Nodes (6): _dash(), FakePusher, test_prune_is_opt_in(), test_restart_skips_files_already_hosted(), test_round_pushes_changes_once(), test_selects_data_files_only()

### Community 43 - "JEPA Environment Config"
Cohesion: 0.20
Nodes (9): CUDA_VISIBLE_DEVICES, HF_HOME, PYTHONUNBUFFERED, SCIJEPA_CHECKPOINT_ROOT, SCIJEPA_DASHBOARD_ROOT, SCIJEPA_DATA_ROOT, SENTENCE_TRANSFORMERS_HOME, JEPA-rea.sh script (+1 more)

### Community 44 - "LLM Rebuttal Script"
Cohesion: 0.20
Nodes (9): HF_HOME, MPLBACKEND, MPLCONFIGDIR, OMP_NUM_THREADS, PYTHONPATH, PYTHONUNBUFFERED, SENTENCE_TRANSFORMERS_HOME, run_LLM_rebuttal_w126.sh script (+1 more)

### Community 45 - "How it works / why it is true"
Cohesion: 0.13
Nodes (14): How it works / why it is true, How you switch it on: the config, In one sentence, `manual_backward`, Manual optimization: `automatic_optimization = False`, Questions you may get, and good answers, Slide 17 — With vs without Lightning: the same step, wired through hooks, The `devices > 1` row (+6 more)

### Community 46 - "Brief: build the "Paper A training optimization" deck (pptx + pdf)"
Cohesion: 0.07
Nodes (26): 10. Slide-by-slide plan, 11. Design notes, 12. Files in `paperA_H100_results_2026-10-24.zip`, 1. Audience, goal, constraints, 2. The story in one paragraph, 3. Setup (one slide), 4. What changed (one slide, ideally with a small diagram), 5.1 Time & efficiency: step time (ms, lower is better) (+18 more)

### Community 47 - "Background you need"
Cohesion: 0.12
Nodes (17): Bottom note, Far right: a light-blue box "added to QKᵀ/√d in SDPA", Formula box (grey, monospace), How it works / why it is true, How the old and new code build this grid (preview of slides 5 and 6), In one sentence, Left: `rel[b] · T × T, uint8`, Middle: `W · 22 types × 8 heads` (+9 more)

### Community 49 - "ku_graph.py"
Cohesion: 0.14
Nodes (12): 6.3 `data/ku_graph.py`, Stand-ins for data that does not exist yet, build_proxy_edges(), build_blob(), build_context_texts(), iter_records(), iter_selected(), keep_record() (+4 more)

### Community 52 - "Dataset Split Logic"
Cohesion: 0.18
Nodes (11): _as_text(), ku_fix_drops(), key(), main(), make_split(), parquet_files(), parse_args(), read_rows() (+3 more)

### Community 53 - "experiment.py"
Cohesion: 0.07
Nodes (38): _context(), _dashboard(), main(), _tensorboard(), _train(), set_cfg(), _atomic(), dashboard_root() (+30 more)

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
Cohesion: 0.30
Nodes (10): corpus_html(), _edge(), _hub_node(), instance_html(), _ku_node(), paper_html(), short_id(), _snip() (+2 more)

### Community 58 - "_RelBias"
Cohesion: 0.18
Nodes (4): What the real code does, rel_bias_heads(), _RelBias, _RelBiasHeads

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
Cohesion: 0.19
Nodes (11): _agg(), _by_kind(), _consistency_probe(), feats(), _intervention(), _pdom(), _rank_instances(), text_baseline() (+3 more)

### Community 71 - "Evaluation Pipeline"
Cohesion: 0.10
Nodes (14): task_summary(), board(), _bool(), display(), domain_balance_table(), _env(), epoch_table(), epoch_types() (+6 more)

### Community 72 - "Sampler"
Cohesion: 0.11
Nodes (6): 6.5 `data/instances.py`, Sampler, data(), mask_plan(), plan(), split()

### Community 73 - "Inference and Embedding"
Cohesion: 0.09
Nodes (17): Benchmark validity (validity.py), Data split, Diagnostics, always reported together, Knowledge units (KUs), KU modes, Leak-safe masking, Main findings (findings.py), Masking (+9 more)

### Community 74 - "report.py"
Cohesion: 0.12
Nodes (20): compute(), figures(), timeline(), hardware(), infra_pages(), llm_figure(), llm_pages(), _llm_rows() (+12 more)

### Community 75 - "instances.py"
Cohesion: 0.13
Nodes (11): _add_text_token(), build_instance(), _drop_tokens(), _extra_anchors(), _hard_negative(), _leak_family(), motif_of(), _paper_tokens() (+3 more)

### Community 76 - "Validity and Faithfulness"
Cohesion: 0.13
Nodes (15): 9.4 Other evaluation modules, agreement(), apply_filter(), audit(), _cell(), contamination(), export(), faithfulness() (+7 more)

### Community 77 - "Training Telemetry"
Cohesion: 0.15
Nodes (3): _atomic_dump(), _clean(), LiveRun

### Community 78 - "Evaluation CLI"
Cohesion: 0.15
Nodes (14): cmd_model(), cmd_results(), cmd_system(), fmt(), g(), get_gi(), headline(), load_cfg() (+6 more)

### Community 79 - "Diagnostics Evaluation"
Cohesion: 0.05
Nodes (43): 16. Extending the code, 8.1 `train_one(cfg, gi, seed, log, live, save_dir, ckpt_path)`, 9.1 Main evaluation: `evaluation/diagnostics.py`, 9.2 Reasoning tasks: `evaluation/tasks.py`, ablate_structure(), add_action(), build_view(), collate() (+35 more)

### Community 80 - "Parquet Test Suite"
Cohesion: 0.13
Nodes (4): json_files(), parquet_files(), _record_files(), TestEndToEnd

### Community 82 - "serve.py"
Cohesion: 0.21
Nodes (3): _Quiet, register_graph(), start_server()

### Community 83 - "SciKU Bench Paper Guide"
Cohesion: 0.10
Nodes (19): 10The A\* paper, 11What reviewers will ask, 12Work plan, 1Short answers, 2The five reasoning tasks, 3The key finding: in their current form, three tasks are solved without reasoning, 4Making the tasks require reasoning, 5Why no-training baselines are required (+11 more)

### Community 85 - "PaperAModel"
Cohesion: 0.16
Nodes (8): What a code graph is, gpu_step_time(), step(), step(), PaperAModel, check_invariants(), _autocast(), _fp32()

### Community 86 - "os"
Cohesion: 0.05
Nodes (26): 5.1 Main groups of `PaperAConfig`, The ICML configuration and experiments, _all_arms(), describe(), method_family(), PaperAConfig, setup_dashboard(), get_index() (+18 more)

### Community 87 - "Neuron Interpretation"
Cohesion: 0.21
Nodes (8): Live dashboard and graph views, _corr(), _label(), _layer_verdict(), neuron_report(), omega2(), _top_group(), _type_centre()

### Community 88 - "SAM Optimizer Module"
Cohesion: 0.07
Nodes (17): ASAM, SAM, flatten_cfg(), GraphJepaLitModule, _batch_size(), covariance_loss(), distance_step(), ema_bounds() (+9 more)

### Community 90 - "How it works / why it matters"
Cohesion: 0.15
Nodes (12): Background you need: git from zero, Bottom row: four rule cards, In one sentence, Left side: the branch diagram, Questions you may get, and good answers, Right side, middle: "From the GitHub web UI" (four numbered rules), Right side, top: "From a terminal" (five lines of commands), Slide 25 — Repository: git rules that keep main runnable for four people (+4 more)

### Community 91 - "bench_compare.py"
Cohesion: 0.10
Nodes (15): D. Software and tools, Hooks, `transfer_batch_to_device`, energy(), log(), nbytes(), _nv(), Probe (+7 more)

### Community 92 - "Record Normalization Tests"
Cohesion: 0.19
Nodes (3): normalize_record(), _safe_filename(), TestNormalizer

### Community 93 - "Knowledge Unit Schema"
Cohesion: 0.08
Nodes (19): Research-domain imbalance, active_templates(), _case(), clean(), compose_text(), counterfactuals(), domain_of(), ku_flags() (+11 more)

### Community 95 - "Hardware Monitor"
Cohesion: 0.23
Nodes (3): Monitor, progress_dir(), write()

### Community 96 - "Project README"
Cohesion: 0.14
Nodes (14): 1. Environment, 2. Data, 3. Run the pipeline, 🙌 Acknowledgements, 💾 Caches, ⚙️ Configuration, ✨ Features, 🚀 Getting Started (+6 more)

### Community 97 - "os"
Cohesion: 0.27
Nodes (5): render_all(), _save(), get(), paired_bars(), savefig()

### Community 98 - "Soft Constraint Loss"
Cohesion: 0.21
Nodes (3): _keep(), ShapeBank, tnorm()

### Community 99 - "System Hardware Info"
Cohesion: 0.28
Nodes (8): collect(), _cpu(), _gpus(), _run(), _slurm(), _storage(), _torch(), _versions()

### Community 100 - "Background you need"
Cohesion: 0.09
Nodes (22): Background you need, Bytes and number formats, How it works / why it is true, In one sentence, Line by line, optimized code (current `models/model.py`, lines 56–91 and 116–123), Line by line, previous code (real code at commit `992779a`, `models/model.py`), masked_fill and the 0 / −∞ mask, permute: changing the labels, not the data (+14 more)

### Community 101 - "Evaluation Protocol"
Cohesion: 0.15
Nodes (13): 1. The data unit: knowledge units, 2. Split by time: no future papers in training, 3. A query: mask one KU, predict it, 4. Ranking against hard candidates, 5. The five reasoning tasks, 6. Comparison systems: same queries, same candidates, 7. Model diagnostics (models only), 8. Seeds, best epoch and hypotheses (+5 more)

### Community 102 - "test_package.py"
Cohesion: 0.16
Nodes (5): all_modules(), test_cli_help(), test_module_imports(), test_paths_come_from_the_environment(), test_smoke()

### Community 103 - "gnn_wrapper.py"
Cohesion: 0.08
Nodes (7): Identity, MLP, VNUpdate, GNN, GATConv, GCNConv, GINEConv

### Community 105 - "CODE_DOCUMENTATION.md"
Cohesion: 0.24
Nodes (4): How compute is counted, Long runs: checkpoints and chained jobs, Outputs for the paper, Training infrastructure

### Community 107 - "Project Setup Guide"
Cohesion: 0.18
Nodes (11): Commands, Configuration, Documentation, Install, Long runs on a cluster (e.g. 60,000 papers), Quick start, Repository layout, Reproducibility (+3 more)

### Community 109 - "SCI-JEPA: technical code documentation"
Cohesion: 0.12
Nodes (16): 12.1 Layout, 12. Outputs and file formats, 13. Reproducibility and determinism, 15. Testing, 17. The 60 k run configuration, 18. Troubleshooting, 1. What the code does, 2. Installation and quick start (+8 more)

### Community 110 - "bench_kit.sh"
Cohesion: 0.29
Nodes (7): HF_HOME, HF_HUB_OFFLINE, PYTHONUNBUFFERED, run(), bench_kit.sh script, TOKENIZERS_PARALLELISM, TRANSFORMERS_OFFLINE

### Community 111 - "Background you need"
Cohesion: 0.09
Nodes (22): Background you need, "Balanced" MRR, Chance level and "× chance", Hits@1 and Hits@10, How it works / why it is true, In one sentence, Mean and standard deviation (sd), Mean reciprocal rank (MRR) (+14 more)

### Community 112 - "banks.py"
Cohesion: 0.12
Nodes (10): bm25_ranks(), BM25Index, _fingerprint(), get_bm25_index(), hard_pool(), pooled_ranks(), strip_ngram_overlap(), _tok() (+2 more)

### Community 113 - "Slide 7 — Why fewer bytes means less time"
Cohesion: 0.10
Nodes (21): Analogy: a kitchen, Arithmetic intensity and the ridge point, Background you need, Effective bandwidth, How it works / why it is true, In one sentence, IO-awareness (FlashAttention), Questions you may get, and good answers (+13 more)

### Community 115 - "Background you need"
Cohesion: 0.10
Nodes (21): Argument 1: the forward pass is bit-identical, Argument 2: the backward pass computes the same sum, Argument 3: uint8 is exact, Argument 4: confirmed end to end, Background you need, fp32 and bf16, GPU atomics and non-determinism, How computers store real numbers: floating point (+13 more)

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
Cohesion: 0.16
Nodes (7): 7.1 Attention with relation biases, 7.2 Encoder families (`make_blocks`), 7. Model, Block, GINEBlock, GPSBlock, make_blocks()

### Community 121 - "Slide 15 — Energy: more power for half as long = −22 to −30% per KU"
Cohesion: 0.10
Nodes (21): Background you need, CO₂e, D0, From J per step to J per 1,000 KUs, GPU only, How energy was measured: the NVML energy counter, How it works / why it is true, In one sentence (+13 more)

### Community 122 - "reporting/figures.py"
Cohesion: 0.33
Nodes (9): export(), cell(), _fig_legend(), _footnote(), _hbar(), _label(), _legend(), _old() (+1 more)

### Community 123 - "SCI-JEPA Overview Docs"
Cohesion: 0.25
Nodes (8): Changing hyperparameters, Checkpoints and resuming, Code graph, Data paths, Install, Logs and the dashboard, SCI-JEPA, Train

### Community 124 - "Training Scaling Report"
Cohesion: 0.33
Nodes (6): 1. Where the time went (before), 2. What was changed, 3. Measured effect (after), 4. Accuracy check (before vs after), 5. Files, Scaling SCI-JEPA training to 60k papers

### Community 125 - "Slide 14 — Change 3, data I/O: relation ids fit in one byte"
Cohesion: 0.10
Nodes (20): 1. 8× fewer bytes for relation ids, 2. −27% total data per step, 3. Why PCIe traffic per second stays about the same, 4. Network on one GPU: none, 5. Network with 4 GPUs: the gradient all-reduce, Background you need, Bytes and integer types, Host and device (+12 more)

### Community 126 - "Slide 16 — Cost: −46% GPU-hours per S run; DDP trades GPU-hours for time"
Cohesion: 0.10
Nodes (20): 28.6 billed H100-hours, 40-epoch run, Background you need, D0 and the extrapolation, "DDP trades GPU-hours for time", GPU-hour, How it works / why it is true, In one sentence (+12 more)

### Community 128 - "Project Documentation"
Cohesion: 0.27
Nodes (6): Agent guide, Commands, graphify, Rules for this repository, Reasoning experiments on the heterogeneous paper graph, `train/` file map

### Community 129 - "Background you need"
Cohesion: 0.11
Nodes (19): 4-GPU numbers, A GPU does two kinds of work, Background you need, FLOP and FLOPs, FLOP/s and TFLOP/s, "GPU busy" (GPU utilisation), Host CPU, How it works / why it is true (+11 more)

### Community 130 - "Slide 19 — DDP: averaging per-GPU gradients is exactly the full-batch gradient"
Cohesion: 0.11
Nodes (19): A tiny numeric example (B = 4, W = 2), Background you need, DDP and all-reduce, How it works / why it is true, In one sentence, Questions you may get, and good answers, Ring all-reduce, as a picture in words, Slide 19 — DDP: averaging per-GPU gradients is exactly the full-batch gradient (+11 more)

### Community 135 - "data_utils/papers.py"
Cohesion: 0.32
Nodes (3): _coerce_summary(), _extract_node_texts(), parse_paper()

### Community 136 - "Contributing Guidelines"
Cohesion: 0.29
Nodes (7): Branches, Code graph, Contributing, Pull requests, What not to commit, Working from a terminal, Working from the GitHub web UI

### Community 137 - "Slide 2 — The answer: half the GPU time, same arithmetic, same MRR"
Cohesion: 0.11
Nodes (18): Background you need, Card 1: 1.84× faster, and why the title says "half", Card 2: −27% energy per KU, although power went up, Card 3: 0.904 vs 0.905, "the same", Card 4: 0 FLOPs changed, Chip 1: Lightning on 1 GPU = same computation, Chip 2: 4 GPUs, 2.1–3.0× sooner, but more GPU-hours and energy, Chip 3: one H100 per run (+10 more)

### Community 138 - "Architecture Docs"
Cohesion: 0.33
Nodes (6): Architecture, Generated views (Graphify), Graph-JEPA (`Analysis/`), How the pieces connect, Paper A (`SCI-JEPA-training/`), The papers pipeline (`Analysis/JEPA-rea.sh`)

### Community 139 - "Slide 13 — Memory: the honest cost is +20–37% reserved memory"
Cohesion: 0.11
Nodes (18): Allocated vs reserved: the caching allocator, Background you need, Fix if ever needed: bf16 one-hot, GPU memory, How it works / why it is true, In one sentence, Is it a problem? No, here, Questions you may get, and good answers (+10 more)

### Community 140 - "Slide 22 — Which GPU? H100 measured, A100 and L40S pending"
Cohesion: 0.11
Nodes (18): Background you need, How it works / why it is true, How `regenerate.sh` fills the cells later, In one sentence, NVLink vs PCIe, and the 4-GPU DDP row, Questions you may get, and good answers, Ridge points, computed (derived from datasheet values), Roofline in two sentences (Williams, Waterman & Patterson, CACM 2009) (+10 more)

### Community 141 - "Slide 8 — Evidence: the time disappears exactly in the changed kernels"
Cohesion: 0.12
Nodes (17): Background you need, How it works / why it is true, In one sentence, Kernel time versus step time, Questions you may get, and good answers, Shares of the step, Slide 8 — Evidence: the time disappears exactly in the changed kernels, The arithmetic (+9 more)

### Community 142 - "Slide 11 — Time: 1.6–1.9× faster per GPU; 4 GPUs a further 2.1–3.0×"
Cohesion: 0.12
Nodes (17): Background you need, Evaluation time (notes only), How it works / why it is true, In one sentence, Questions you may get, and good answers, Slide 11 — Time: 1.6–1.9× faster per GPU; 4 GPUs a further 2.1–3.0×, Speed-up, Step time (+9 more)

### Community 143 - "Slide 24 — Repository: Graphify turns the code into a queryable map"
Cohesion: 0.12
Nodes (16): AST-only extraction, no LLM, Background you need, Bottom row: "Also added" (five small cards), Git hooks, How it works / why it matters, In one sentence, MCP, hooks and skills for AI assistants, Middle row: three consumers of the same graph (+8 more)

### Community 147 - "Slide 18 — With vs without Lightning, measured on 1 H100: no difference"
Cohesion: 0.13
Nodes (15): Background you need, Every row of the Δ table, with the arithmetic, How it works / why it is true, In one sentence, Questions you may get, and good answers, Slide 18 — With vs without Lightning, measured on 1 H100: no difference, The "adds" and "costs" boxes, What to say when presenting (+7 more)

### Community 148 - "T"
Cohesion: 0.25
Nodes (14): box(), card(), chip(), circleNum(), code(), ex(), gain(), leaf() (+6 more)

### Community 149 - "BM25System"
Cohesion: 0.16
Nodes (5): BM25System, _consistency_unsup(), _needed_ids(), _pairs(), _split_ids()

### Community 150 - "stages_sim.py"
Cohesion: 0.21
Nodes (7): spectral_decay_kappa(), _collapsed(), _critical_rho(), collapsed_at(), _measured_rho(), _simulate(), stage_phase()

### Community 151 - "Slide 1 — Paper A trains 1.8× faster at the same accuracy"
Cohesion: 0.15
Nodes (13): Background you need, How it works / why it is true: the whole story of the deck, In one sentence, Questions you may get, and good answers, Slide 1 — Paper A trains 1.8× faster at the same accuracy, What does "1.8× faster" mean?, What does "same accuracy" mean?, What is an H100? (+5 more)

### Community 152 - "03-setup.md"
Cohesion: 0.15
Nodes (12): 1. "Variants": four coloured pills, 2. "Data": a horizontal bar split in three, 3. "Models (Paper A sizes)": three cards, 4. "Hardware and instruments": a table, Background you need, How it works / why it is true, In one sentence, Questions you may get, and good answers (+4 more)

### Community 153 - "Slide 20 — Scaling: 4 independent runs 3.99× per node, one DDP run 2.06×"
Cohesion: 0.15
Nodes (13): Background you need, DDP speed-up and efficiency for three configurations, How it works / why it is true, In one sentence, Questions you may get, and good answers, Slide 20 — Scaling: 4 independent runs 3.99× per node, one DDP run 2.06×, The node-throughput numbers, with arithmetic, The three reasons the DDP run is below 4× (+5 more)

### Community 154 - "Slide 21 — How many H100s? One per run, four only when time matters"
Cohesion: 0.15
Nodes (13): Background you need, How it works / why it is true, In one sentence, Questions you may get, and good answers, Slide 21 — How many H100s? One per run, four only when time matters, Summary table, The "no" branch: one H100 per run, four runs per node, The "yes" branch: one run on 4 H100s (+5 more)

### Community 156 - "Glossary: every technical word in the deck, in plain language"
Cohesion: 0.17
Nodes (12): A. Hardware: GPUs, memory, connections, B. Data, tensors and number formats, C. The model: attention, JEPA, training, Contents, E. Performance: time, compute, memory traffic, F. Multi-GPU training, G. Accuracy and statistics, Glossary: every technical word in the deck, in plain language (+4 more)

### Community 157 - "encoder_system"
Cohesion: 0.28
Nodes (6): 9.3 Comparison systems: `evaluation/baselines.py`, _encode_chunked(), encoder_system(), make_system(), SupervisedSystem, text_system()

### Community 158 - "probe.py"
Cohesion: 0.16
Nodes (8): collect_features(), probe_classification(), probe_regression(), collapse_diagnostics(), _effective_rank(), _ema_range(), _fused(), _per_group_std()

### Community 160 - "check_environment.py"
Cohesion: 0.36
Nodes (3): page(), step(), free_port()

### Community 161 - "Slide 23 — Repository: one Lightning runtime in each pipeline"
Cohesion: 0.22
Nodes (8): How it works / why it matters, In one sentence, Questions you may get, and good answers, Right column: "Paper A · SCI-JEPA-training/", Slide 23 — Repository: one Lightning runtime in each pipeline, What to say when presenting, What you see on the slide, Words to remember

### Community 162 - "Background you need"
Cohesion: 0.22
Nodes (9): Backend, Background you need, Environment variables, GPU, DDP, rank, seed, PyTorch Lightning, Repository, folder, file, function, SLURM, cluster, job, Training loop (+1 more)

### Community 163 - "sci_jepa/cli.py"
Cohesion: 0.29
Nodes (5): fetch_assets(), main(), usage(), ensure_vendor(), test_cli_lists_every_command()

### Community 164 - "assemble.mjs"
Cohesion: 0.18
Nodes (6): dash, here, infra, out, pkg, sitenav

### Community 165 - "train"
Cohesion: 0.33
Nodes (5): covariance_loss(), prediction_loss(), test(), train(), variance_loss()

### Community 166 - "NVIDIA H100"
Cohesion: 0.25
Nodes (7): local b64 fast, local b64 gpu, M b128 gpu, NVIDIA H100, Previous vs optimized Paper A code, S b128 gpu, S b256 gpu

### Community 167 - "Background you need"
Cohesion: 0.29
Nodes (7): Background you need, What a matrix multiplication (GEMM) is, What a one-hot matrix is, What G is, What "loop-invariant code hoisting" means, What "training" and a "gradient" are, Why the gradient of a lookup is a "scatter-add"

### Community 168 - "Slide 6 — Change 2, backward: build the loop-invariant one-hot once"
Cohesion: 0.17
Nodes (12): A tiny worked example (4 token pairs, 2 relation types, 1 head), How it works / why it is true, In one sentence, Questions you may get, and good answers, Slide 6 — Change 2, backward: build the loop-invariant one-hot once, The honest cost, What to say when presenting, What was measured (+4 more)

### Community 170 - "rect"
Cohesion: 0.33
Nodes (6): bar(), gauge(), grid(), op(), rect(), row()

### Community 171 - "has"
Cohesion: 0.60
Nodes (5): has(), meas(), mm(), spRow(), stepRow()

### Community 172 - "compilerOptions"
Cohesion: 0.18
Nodes (10): compilerOptions, lib, module, moduleResolution, noEmit, skipLibCheck, strict, target (+2 more)

### Community 173 - "regenerate.sh"
Cohesion: 0.40
Nodes (4): NODE_PATH, PATH, PPTX_SKILL, regenerate.sh script

### Community 174 - "6. Data layer"
Cohesion: 0.40
Nodes (5): 6.1 `data/ku_schema.py`, 6.2 `data/split.py`, 6.4 `data/graph_index.py`, 6. Data layer, paper_year()

### Community 175 - "How it works / why it matters"
Cohesion: 0.20
Nodes (10): graph.json: its own merge driver, and hooks, How it works / why it matters, Keep paths stable, Long-lived branches, Never commit data or outputs, The branch prefixes, The four web UI rules, and why each exists, The terminal workflow, line by line (+2 more)

### Community 177 - "Explanations for the deck "Paper A trains 1.8× faster at the same accuracy""
Cohesion: 0.50
Nodes (4): Explanations for the deck "Paper A trains 1.8× faster at the same accuracy", Reading order, The story in plain words, Where the facts come from

### Community 178 - "11. Running on a Slurm cluster"
Cohesion: 0.33
Nodes (5): `on_train_batch_end`, rank 0 and `broadcast(stop)`, 11.1 Scripts, 11.2 One job in `dist` mode, 11.3 Dashboard access, 11. Running on a Slurm cluster

### Community 179 - "Migration from SCI-JEPA-training/Analysis"
Cohesion: 0.50
Nodes (4): Changed after the migration (intended), Migration from SCI-JEPA-training/Analysis, What changed for users, Where things went

### Community 180 - "InverseRelationHead"
Cohesion: 0.20
Nodes (3): GraphEncoder, InverseRelationHead, rwse()

### Community 181 - "seg"
Cohesion: 0.67
Nodes (3): arrow(), line(), seg()

### Community 182 - "Background you need"
Cohesion: 0.22
Nodes (9): Attention, from scratch, Attention heads, Background you need, FLOPs and bytes, Lookup table (embedding), Shapes, The mask, Tokens and sequences (+1 more)

### Community 183 - "hosted.js"
Cohesion: 0.47
Nodes (4): ago(), manifest(), paintSynced(), toLogin()

### Community 186 - "Background you need"
Cohesion: 0.40
Nodes (5): Background you need, LightningModule, `trainer.fit`, What is a "training loop"?, What is PyTorch Lightning, and why do people use it?

### Community 187 - "Hosted dashboard (Vercel)"
Cohesion: 0.40
Nodes (4): Deploy and settings, Hosted dashboard (Vercel), How data gets there, Run the sync on the cluster

### Community 190 - "iter_stored"
Cohesion: 0.40
Nodes (3): iter_stored(), load_manifest(), split_caps()

### Community 192 - "vercel.json"
Cohesion: 0.50
Nodes (3): headers, rewrites, $schema

## Knowledge Gaps
- **76 isolated node(s):** `@types/node`, `typescript`, `HF_HOME`, `HF_HUB_OFFLINE`, `PYTHONUNBUFFERED` (+71 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1504 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **45 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `train_one()` connect `Training Loop Execution` to `Slide 2 — The answer: half the GPU time, same arithmetic, same MRR`, `Slide 11 — Time: 1.6–1.9× faster per GPU; 4 GPUs a further 2.1–3.0×`, `Slide 18 — With vs without Lightning, measured on 1 H100: no difference`, `Slide 1 — Paper A trains 1.8× faster at the same accuracy`, `03-setup.md`, `Glossary: every technical word in the deck, in plain language`, `_batches`, `Slide 23 — Repository: one Lightning runtime in each pipeline`, `Background you need`, `How it works / why it is true`, `Brief: build the "Paper A training optimization" deck (pptx + pdf)`, `experiment.py`, `Evaluation Pipeline`, `Sampler`, `Validity and Faithfulness`, `Evaluation CLI`, `Diagnostics Evaluation`, `PaperAModel`, `os`, `Neuron Interpretation`, `bench_compare.py`, `Soft Constraint Loss`?**
  _High betweenness centrality (0.190) - this node is a cross-community bridge._
- **What connects `@types/node`, `typescript`, `HF_HOME` to the rest of the system?**
  _76 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `auth.ts` be split into smaller, more focused modules?**
  _Cohesion score 0.058445353594389245 - nodes in this community are weakly interconnected._
- **Why does `Background you need` connect `03-setup.md` to `Training Loop Execution`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Should `Paper Reasoning JEPA` be split into smaller, more focused modules?**
  _Cohesion score 0.08826945412311266 - nodes in this community are weakly interconnected._
- **Why does `Words to remember` connect `03-setup.md` to `Training Loop Execution`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Should `Link Prediction Model` be split into smaller, more focused modules?**
  _Cohesion score 0.07307692307692308 - nodes in this community are weakly interconnected._