# Paper A — Constraint-Grounded Latent Prediction (Track A code)

Self-contained package. It reads the normalized Scientific-Summaries parquet directly
(`Scientific-Summaries-Data/normalized/<subset>/*.parquet`, the 44-key schema written by
`parquet_to_json.py`) and depends on nothing else in the repo.

```
python -m train.paper_a.smoke_test                              # CPU, synthetic records, ~30 s
python -m train.paper_a.ku_graph --set max_papers=60000         # build + cache the D0 KU graph
python -m train.paper_a.train --audit-shapes                    # Theorem A2 audit, no training
python -m train.paper_a.train --exp E10 --size S                # one ablation, 5 seeds, paired tests
python -m train.paper_a.train --size S --set n_subgraphs=8 lambda_inv=0.3   # one config
sbatch run_paper_a.sh                                           # cluster: audit + E10 E11 E12 E4
```

`--set` takes any field of `PaperAConfig` (`config.py`). Results go to
`$PAPER_A_OUT/<exp>/`: one JSON per arm and seed, plus `summary.json` (mean, std, and a paired
t-test against the first arm on every key metric).

## Live dashboard and graph views

`telemetry.py` streams each run to `<dashboard>/runs/<name>/live.json`. `dashboard.html`
(copied to `<dashboard>/index.html` by `serve.setup_dashboard`) polls it every 2 s.
`graph_viz.py` writes animated vis-network pages (paper subgraph, training instance step by
step, corpus graph). `serve.py` serves the folder on localhost, because browsers block live
fetches from `file://`. Enable streaming with
`train_one(..., live=LiveRun(root, name, cfg, gi))` or `run_experiment(..., live_root=root)`;
set `eval_every` for live validation curves. With `neuron_report` (default on) each validation
pass also runs `neurons.py`, which labels every encoder neuron and predictor output by its
bias-corrected variance share (ω²) for KU type, paper and field and by its within-type
correlation with the paraphrase-leak flag: dead / identity / type / field / leak / content.
A final report runs on the test split. Dashboard tabs: Overview, Explain (rule-based
plain-language findings plus glossary), Training dynamics, Validation, Neurons, Where it learns,
Examples, Graphs, All runs; the x-axis is epoch or step. `paper_a_local.ipynb` wires all of this together.

## Saved models and inference

Every training run writes a model folder (`save_model`, on by default): the single notebook run to
`paper_a_results/models/<run>/`, experiment runs to `paper_a_results/<exp>/models/<arm>_seed<s>/`.
It holds `model.safetensors` (end of training, every module including the EMA target encoder),
`best.safetensors` (best end-of-epoch validation), `config.json` (the full configuration, so the
identical model is rebuilt) and `summary.json` (test metrics, text encoder, data fingerprint).
`inference.SciJEPA.from_pretrained(folder)` loads it with the same frozen text encoder and offers
`embed` (facet embeddings), `predict` (hidden-part representation, optionally under a reasoning
template), `rank` (candidate texts for a hidden part) and `consistency` (supports vs contradicts) for
any paper given as a structured summary. The paper is encoded on its own (no neighbours) and `rank`
encodes candidates alone, a lighter version of the evaluation protocol.

## KU modes

| `ku_mode` | The hidden unit | How the model sees the hidden unit's own paper |
|---|---|---|
| `section` (default) | one section / claim / evidence / implication KU | the paper's other KUs as separate tokens (leak-safe masking) |
| `paper_masked` | the same KU as `section` | ONE text made of the visible KU types, joined and encoded together |
| `paper` (not in the paper) | a whole paper | not at all: predicted from neighbouring papers only |

`paper_masked` uses query templates (`ku_schema.TEMPLATES`): the three reasoning queries (forward,
inverse, abduction) and `all-<t>` = everything except `t` and the KUs that restate it
(`LEAK_FAMILY`). Training draws a random template that hides the target type; evaluation uses
`all-<t>` and the task templates. The hidden KU, its candidate pool and the neighbouring papers
are identical to `section`, so **E17** (section vs paper_masked) isolates one question: is it
better to read a paper as one text or as a graph of separate KUs? Each paper needs ~13 extra
encodings (`pm_templates="full"`) or 3 (`"tasks"`). For interventions, every val / test paper also
gets counterfactual forward texts with the procedure or method of up to 3 donor papers.
The dashboard's **Compare** tab shows two runs side by side (default: the arms of E17).

## Reasoning-task evaluation matrix

`tasks.py` queries the trained model five ways, at the end of every epoch (validation) and at the
end of training (test). It is on by default (`eval_tasks`, `task_eval_instances`). The parquet has
no separate Problem / Material / Condition fields, so the slots are mapped to KU types:
Problem = context + question, Method = method, Material / Condition = procedure, Result = result,
Claim / Hypothesis = claim (+ question). "given" = only these KU types stay visible in the target's
own paper; neighbouring papers are unchanged.

| Task | Query | Score |
|---|---|---|
| Forward prediction | mask result \| given problem, method, procedure | balanced MRR ÷ chance |
| Experiment design (inverse) | mask method / procedure \| given problem, result | balanced MRR ÷ chance |
| Abduction | mask claim / question \| given result, procedure | balanced MRR ÷ chance |
| Consistency checking | hide claim–evidence edges \| inverse head: supported_by vs challenged_by | balanced accuracy (chance 0.5) |
| Intervention conditioning | forward query with the procedure (condition) or method swapped for a donor paper's | relative shift of the prediction; share moving toward the donor's result |

Limits, stated in the output: there are no supports / contradicts edges *between* papers in the
data, so consistency is within-paper only; the model has no `z_do` input, so interventions are
input swaps. Not applicable with `ku_mode="paper"`. With `paper_masked` the given parts are the text the model reads, and interventions re-encode that text with the donor's section. Dashboard: *Reasoning tasks* tab, *All runs* →
"Reasoning tasks: every run".

## Text encoder

KU node features come from a frozen sentence encoder chosen by `config.embedder` (presets in
`config.EMBEDDERS`). The default, `gte-modernbert` (Alibaba-NLP/gte-modernbert-base, 768-d),
has an 8,192-token context and `embed_max_tokens = 2048`. The longest KU measured is ~1,150
tokens (a `result`), so no KU is truncated. The old `minilm` (256 tokens) cut 93% of method, 85% of
procedure and 78% of result KUs. Cosine scales differ between encoders (two random KUs:
gte-modernbert ~0.53, MiniLM ~0.08), so each preset carries its own `proxy_tau` / `audit_tau`.
For gte-modernbert these are 0.86 / 0.87, the same percentiles as MiniLM's 0.80 / 0.75, measured on
600 real papers. The leak bins in the evaluation are relative to `audit_tau`. The index build
prints how many texts were truncated.

## Knowledge units (KUs)

Defined in `ku_schema.py`, measured on arxiv-new, arxiv-old and titled shards (~23 KUs per paper):

| KU type | Source | Per paper |
|---|---|---|
| context, question, method, procedure, result, interpretation, limitation | `research_context`, `research_question_hypothesis`, `methodological_details`, `procedures_architectures`, `key_results`, `interpretation_implications`, `contradictions_limitations` | 1 each (~100% filled, 900–1,600 chars) |
| claim | `claims[].description` (`claims` is a JSON string) | ~4.3 (3–6) |
| evidence | `claims[].supporting_evidence`, `claims[].contradicting_evidence` (the edge says which) | ~4.3 supporting + ~1.5 contradicting |
| implication | `claims[].implications` | ~4.3 |
| figures, robustness, data_code, ethics (off by default) | `key_figures_tables`, `robustness_ablation_notes`, `data_code_availability`, `ethical_considerations` | sparse, placeholder-heavy |

- **Never KUs:** `executive_summary` and `three_takeaways` restate the units we mask.
- **Cleaning:** missing values, placeholders ("N/A", "Not applicable.", "No links provided.") and
  text under `ku_min_chars` are not KUs. `contradicting_evidence` is filtered sentence by sentence
  for null statements ("The paper does not present any contradicting evidence." and ~50
  variants), keeping any real caveat after them. On real shards: 36% of claims have real
  counter-evidence, 42% have none, 18.5% only a null statement, 3.4% a placeholder. Per-type
  outcomes are saved in `build_stats.ku_outcomes`.
- **Intra-paper role edges are imposed by the schema**, not extracted:
  context→question→method→{procedure, result}, result→claim, claim→{evidence, implication},
  result→{interpretation, limitation}. The paper must describe them that way.
- **Paper metadata:** `field_subfield` is split into a coarse field (hub embedding) and a fine
  subfield (hard negatives, neighbourhood fill). Year comes from `source_year`, then `oa_year`,
  then the arXiv id. Citations come from `openalex_id` + `oa_referenced_works`; only ~22% of
  papers have references, so neighbourhoods are filled from the same subfield.
- **Tiers:** D0 60k, D1 500k, D2 2M papers, sampled as `rows_per_shard` random rows from shuffled
  shards. Each shard is one subset and a narrow year range, so taking whole shards would break the
  time split.

## Masking

All 10 core KU types are masked (`target_types`), mixed in every batch (`mask_schedule="mixed"`).
Hidden KUs are drawn without replacement, so one epoch masks every KU exactly once; `epochs`
(default 10) sets the run length. Leak rules: a masked claim hides its evidence and
implications, a masked evidence or implication hides its claim, and a masked method hides its
procedure (and vice versa). E15 / E18 compare this one model with one model per masked type.

The code also has `balanced`, `round_robin` and `sequential` (+ `replay_frac`) schedules. They are
not in the paper: on 500 papers, sequential masking with one epoch never left the average
prediction (1.0x chance), while mixed masking with 10 epochs reached 8.8x chance.

## Objective

```
L = mean_d ||Pred(E(x), z_tau) - sg(E_bar(y))||^2      (per-dimension MSE, so lambdas are size-independent)
  + lambda_shacl L_SHACL + lambda_align L_align + lambda_reg L_reg
  + lambda_inv L_inverse + lambda_contrast L_contrast
```
Every λ defaults to 0, so the reference arm is the plain objective. `L_inverse` and `L_align`
make the model a hybrid rather than a pure JEPA, and the paper should say so.

## The ICML configuration and experiments

Every run uses the `PaperAConfig` defaults (`config.ICML_FIXED` describes them): section KUs,
gte-modernbert, time split T = 2018, all 10 KU types mixed, leak-safe masking, N = 8 papers,
plain JEPA objective, 10 epochs, evaluation every epoch plus the reasoning-task matrix. Each
experiment changes one thing (`config.PAPER_EXPERIMENTS`, in paper order):

| Exp | Arms | Question | Status |
|---|---|---|---|
| **BENCH** | text baseline (no training) / reference / paper-as-text / world model / motif / motif + world / Graph-JEPA + world | The benchmark: every main variant and the frozen-text baseline on identical papers, pools and evaluation; ranked in the dashboard's Benchmark tab | runnable |
| E10 | N ∈ {1, 4, 8, 16} | Widening the context removes the A1 degeneracy | runnable |
| E11 | λ_inv ∈ {0, .1, .3, 1} | Inverse-relation head as an order-one barrier | runnable |
| E12 | regulariser off / within / across subgraph | H-JEPA Prop. 2, transposed to the subgraph axis | runnable |
| E4 | resample × census | Repair mechanism one: graph construction | runnable |
| E5 | contrast off / on | Relational contrast masking | runnable |
| E15 | mixed + one model per KU type (section) | Which KU types are learnable, and does one model for all help? | runnable |
| E17 | section vs paper_masked | Read a paper as separate KUs or as one text of its visible parts? | runnable |
| E18 | mixed + one model per KU type (paper_masked) | E15 with the paper read as one text | runnable |
| E19 | reference / world model / world model + inverse head | Does action-conditioned prediction teach structure without losing accuracy? | runnable |
| E20 | reference / RWSE / motif / both / both + world model | Do Graph-JEPA's sub-graph targets and structural encodings make the model use structure? | runnable |
| E1 | SHACL off / loose / tight | Theorem A2 classification | **placeholder shapes** |
| E2 | product / Łukasiewicz / Gödel × depth 3, 8, 15 | Proposition A3 depth decay | **placeholder shapes, replicated conjuncts** |
| E3 | factored vs single head × depth | Shape-factored heads | **placeholder shapes** |

**BENCH, the benchmark.** One code runs the no-training text baseline (`baselines.py`: the mean
frozen embedding of the hidden KU's visible paper-mates ranked by cosine against the same 100
candidates, with the same leak-safe masking and task templates) and every main model variant. The
dashboard's Benchmark tab ranks the arms (mean ± std over seeds), marks the best value per column,
and shows the gain over the reference and whether each arm beats the text baseline. On the
500-paper pilot the text baseline reached about 9.6x chance and the best trained arm 9.5x, so
beating it at 5,000+ papers is the result that justifies the model.

**E19, the JEPA world model.** A structure test on the 500-paper pilot showed that the reference
model's 8.8x-chance accuracy does not depend on typed edges at all: removing every role, citation,
field and proxy edge left it at 8.8x. The learned relation biases were large only for
`same_paper` / `none`. E19 adds action-conditioned prediction: a quarter of training instances
get an action token (`do_swap`: a KU replaced by another paper's of the same type; `do_flip`:
a supports <-> contradicts edge flipped), linked by the `do` relation to what it changes, and the
target is the hidden KU's latent in the paper AFTER the action. The plain objective is unchanged
and accuracy is always measured on clean instances. `worldview.py` reports structure reliance
(x chance with vs without typed edges), action prediction (direction cos between predicted and
true change) and explained example subgraphs, shown in the dashboard's World model tab.

**E20, Graph-JEPA ideas** (Skenderi et al., *Graph-level representation learning with joint-embedding
predictive architectures*). Two of its components are adapted to the KU graph:
(1) sub-graph targets: in half of the training instances the hidden KU's whole motif is hidden and
predicted (a claim with its evidence and implications; method + procedure; result + interpretation +
limitation; context + question). The hidden nodes keep the links among themselves (the motif's
shape) and lose their links to visible KUs. (2) Random-walk structural encodings (RWSE, 16 steps on
the graph of typed links) are added to every encoder token, and the predictor gets the hidden
nodes' true structural position, as in Graph-JEPA's predictor (their Eq. 5). Graph-JEPA's 2-D
hyperbola target is not used: it compresses each target to one number, which would remove KU
retrieval. Motifs are used in training only, and evaluation is the reference protocol, so accuracy
compares one to one with the reference arm. New layers have their own seeds, so the reference
initialisation is unchanged.

Removed from the experiment list (not in the paper): the masking-schedule comparison, one model
per type without the mixed reference (now E15), and one-KU-per-paper mode (`ku_mode="paper"`,
which hides whole papers, not parts). Their code paths remain and are smoke-tested.

Every arm of an experiment starts from identical encoder and predictor weights: optional modules
(shape heads, the census embedding) use their own seed. The smoke test checks this.

## Diagnostics, always reported together

- **chi:** within-subgraph variance share after removing type centroids (H-JEPA Def. 1 scale).
  `chi_raw` (with type centroids) and `chi_per_type` are reported alongside.
- **Partition probe:** linear probe for subgraph identity, split within type, plus the
  between-subgraph variance share. G1 wants the share below 30%.
- **Retrieval:** MRR / Hits@1 / Hits@10 against the full same-type bank (standard) and against
  K = `balanced_k` same-type distractors from the same subfield, then proxy / citation
  neighbours, then the same field (balanced). The pool never depends on the sampled
  neighbourhood, so it is identical across E10's N arms. Its mean size and chance MRR are
  reported. The shortcut margin comes with a bootstrap CI.
- **Relation sensitivity:** the relative change in the prediction when one role edge is rewired
  to another subgraph.
- **Inverse accuracy:** the head is trained in every arm. When λ_inv = 0 it is a detached probe
  that cannot affect the encoder.
- **Online detector:** constraint satisfaction on the true target minus satisfaction against
  a same-type impostor.
- **Nuisance audit:** paraphrase similarity between the target and the visible context, the
  fraction of targets whose type plus subgraph fixes them, and the number of papers per instance.

## Stand-ins for data that does not exist yet

Every result must state these. All of them are marked in the code.

| Plan needs | Used here instead | Replace in |
|---|---|---|
| Grounded edges (canonical Material / Method / Metric) | **Proxy:** method KUs with text-embedding cosine ≥ 0.85 across papers | `graph_index.build_proxy_edges` |
| Declared edges (citation stance, replicates, reuses) | OpenAlex `cites` (no stance) between paper hubs | `ku_graph.build_blob` |
| Claim-dependency DAG and its masking regime | not implemented | — |
| ORKG shape vocabulary | `placeholder_shapes.json` (real counter-evidence, evidence cites a figure, quantitative text, paper type, field) | `--set shape_spec=...json`, same format |
| Text encoder older than T | gte-modernbert-base (2024 data): `time_T < 2024` prints a contamination warning | `config.embedder` |
| Confidence weights (E9), z_do / z_pub conditioning | none | `losses.jepa_loss(weights=)`, `model.Predictor` |
| D-ext (OGBN-MAG) | not implemented | `ku_graph.build_blob(records=...)` |

Time-sliced splits are now real (`split_mode="time"`, train ≤ T, val T+1, test T+2; sweep T).

The placeholder shape spec deliberately includes context-determined constraints (field, paper
type, a result's count of grounded claims, which the visible claims reveal) and validity
constraints (cardinality, datatype). Under Theorem A2 these should not separate the degenerate
branch. A claim's counts of supporting evidence and implications are exactly 1 by the extraction
template, so they carry 0 bits and are not slots. `--audit-shapes` checks each constraint's
tight/loose label against the data.

## Leak-safe masking

The default `mask_policy="token+edges+children"` blanks the masked token's features and drops
its role edges. For a claim it also drops the claim's evidence and implication children, which
paraphrase it (supporting evidence usually restates the claim). `+siblings` additionally hides a
masked method's procedure and vice versa, since the two sections restate each other. The smoke
test asserts both. Paper text never becomes a token: each paper is a hub carrying only its field
embedding.
