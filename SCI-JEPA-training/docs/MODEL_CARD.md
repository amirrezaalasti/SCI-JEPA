# Model card: SCI-JEPA (jepa+rgat)

A joint-embedding predictive (JEPA) world model over the knowledge units (KUs) of research papers, with a relation-aware graph encoder (R-GAT). It predicts the representation of a hidden KU (e.g. a result) from the visible KUs and ranks candidate texts; it judges whether evidence supports or contradicts a claim. **It generates no text.**

## Intended use
- Ranking candidate results, methods or claims for a paper, as an aid to a human reader.
- Judging claim-evidence pairs, with human review.
- Research on scientific world models.

## Not permitted
- Decisions about people, papers, grants, admissions, hiring or promotion (this would make it a high-risk system under the EU AI Act, Annex III 3-4).
- Fact-checking, medical, legal or safety decisions without an expert.
- Use outside the training distribution (English arXiv papers, published up to 2024) without checking quality first. **Language: English only.**

## Human oversight
- `consistency(..., abstain_below=t)` returns `decision = "abstain"` when the larger probability is below t. Recommended t = 0.90: answered pairs are right 96 % of the time, 84 % are answered.
- **Do not filter rankings by their score**: the most confident retrieval answers are not more accurate (selective-prediction gain -10 %).

## Training data
- laion/Scientific-Summaries (CC-BY-4.0; LAION e.V. 2025, arXiv:2502.19413), arXiv subset, 60,000 / 5,000 / 5,000 papers, time-based split (test papers 2025-2026).
- KUs come from LLM-written summaries.
- No author, affiliation or e-mail field is read; contact data inside KU texts is redacted in releases.

## Performance by research domain (ratio of balanced MRR to chance, mean of 3 tasks and the seeds)

| domain | × chance | queries per run |
|---|---|---|
| Astrophysics | 15.1 | 29 |
| EESS & Engineering | 14.9 | 28 |
| Physics | 14.9 | 81 |
| Computer Science | 14.6 | 55 |
| Mathematics | 14.5 | 46 |
| Chemistry | 13.2 | 20 |

Domains with fewer than 20 test queries are not shown. Full results with 95 % intervals: the project dashboard and safety report.

## Known limitations (safety checks that fail)
- a reversed copy beats the truth in many questions (edit_error)
- the similarity rule accepts many wrong answers (halluc_on_attempt)
- representations identify the source paper (shortcut) (identity)
- a wrong answer planted in the paper's own text can win (injection_asr)
- planted wrong answers win more often than unrelated planted text (injection_effect)
- a reworded truth does not reliably beat a meaning-reversed edit (meaning_sensitive)
- the strongest kind of meaning-reversing edit fools it (robust_worst)
- meaning-reversing edits are ranked near chance (robustness)
- meaning-reversed copies of the true result often rank above it (robustness_ci)
- confidence does not identify correct answers (selective_gain)
- the typed graph structure barely changes the answers (structure)

Several of these are evaluation-wide properties (e.g. robustness near chance for every system); values with intervals are in the safety report.

## Environmental impact
- Training of the experiment: 49 A100-equivalent hours, about 24 kWh and 9 kg CO2e (A100 400 W at full load, PUE 1.2, grid 0.38 kg CO2e/kWh (German mix)).

## Licence and attribution
- See LICENSE and NOTICE of the release (decision: docs/governance/LICENCE_DECISION.md).
- Cite the dataset: LAION e.V. (2025), Project Alexandria, arXiv:2502.19413.

## Contact, problems and removal requests
- Contact for questions, problems and removal requests: (to be set: an issue tracker URL and a functional e-mail address of the project)
- Removal procedure: docs/governance/REMOVAL_PROCEDURE.md.
