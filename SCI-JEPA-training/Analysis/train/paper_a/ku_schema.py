"""
Knowledge-unit (KU) schema of the normalized Scientific-Summaries parquet
(Scientific-Summaries-Data/normalized/<subset>/*.parquet, 44 canonical keys).

What one record holds (measured on arxiv-new / arxiv-old / titled shards):
  * seven paper-level sections, ~100% filled, ~900-1600 chars each:
      research_context, research_question_hypothesis, methodological_details,
      procedures_architectures, key_results, interpretation_implications,
      contradictions_limitations
  * `claims`: a JSON STRING holding a list of 3-6 objects (mean ~4.3), each
      {description, supporting_evidence, contradicting_evidence, implications}
      -- every sub-field is ONE string; contradicting_evidence is empty in
      ~41% of claims and a placeholder ("None", ...) in another ~3-4%
  * optional sections, often missing or placeholders ("N/A", "Not applicable.",
      "No links provided."): key_figures_tables (~95% filled),
      robustness_ablation_notes (~20-50%), data_code_availability (~15-40%),
      ethical_considerations (~15-45%)
  * paraphrases of the whole paper: executive_summary, three_takeaways.
      NEVER turned into KUs -- they restate the very units we mask.
  * metadata: field_subfield ("Field — Subfield"), type_of_paper (free text,
      e.g. "Theoretical, Methodological"), source_year / oa_year, openalex_id,
      oa_referenced_works (JSON list of OpenAlex URLs, ~22% of papers).

KU types (token types). One KU per section, one `claim` per claim object, one
`evidence` per non-empty supporting / contradicting string (the edge says
which), one `implication` per claim.

Intra-paper role edges are IMPOSED by the schema (the extractor does not emit
them): context->question->method->{procedure, result}, result->claim, claim->
evidence / implication, result->{interpretation, limitation, ...}. They encode
the template's structure, not extracted relations, and must be described as
such in the paper.
"""
import json
import re

# ---------------------------------------------------------------- KU types
SECTION_KUS = {                      # KU type -> parquet column
    "context": "research_context",
    "question": "research_question_hypothesis",
    "method": "methodological_details",
    "procedure": "procedures_architectures",
    "result": "key_results",
    "interpretation": "interpretation_implications",
    "limitation": "contradictions_limitations",
}
OPTIONAL_KUS = {                     # off by default (sparse, placeholder-heavy)
    "figures": "key_figures_tables",
    "robustness": "robustness_ablation_notes",
    "data_code": "data_code_availability",
    "ethics": "ethical_considerations",
}
CLAIM_KUS = ["claim", "evidence", "implication"]
PAPER_KU = "paper"                   # ku_mode="paper": the whole paper as ONE KU
CTX_KU = "context_text"               # ku_mode="paper_masked": the visible part of a paper as ONE text
ACTION_KUS = ["do_swap", "do_flip"]  # world-model action tokens: swap a KU's content / flip an edge's polarity
COMP_TYPES = list(SECTION_KUS) + CLAIM_KUS + list(OPTIONAL_KUS) + [PAPER_KU, CTX_KU] + ACTION_KUS
NEVER_KUS = ("executive_summary", "three_takeaways")   # whole-paper paraphrases
# every core KU type can be a masking target (optional sections excluded: sparse)
ALL_TARGET_TYPES = tuple(SECTION_KUS) + tuple(CLAIM_KUS)

# ---------------------------------------------------------------- relations
# 0 none, 1 same paper, 2 hub membership, role relations, then hub-hub ones.
REL_NONE, REL_SAME_PAPER, REL_MEMBER = 0, 1, 2
SCHEMA_EDGES = [                     # (src type, relation, dst type)
    ("context", "motivates", "question"),
    ("question", "addressed_by", "method"),
    ("method", "implemented_by", "procedure"),
    ("method", "produces", "result"),
    ("result", "grounds", "claim"),
    ("claim", "supported_by", "evidence"),
    ("claim", "challenged_by", "evidence"),
    ("claim", "implies", "implication"),
    ("result", "interpreted_by", "interpretation"),
    ("result", "limited_by", "limitation"),
    ("result", "shown_in", "figures"),
    ("result", "stress_tested_by", "robustness"),
    ("procedure", "released_as", "data_code"),
    ("method", "governed_by", "ethics"),
]
ROLE_RELS = [r for _, r, _ in SCHEMA_EDGES] + ["proxy"]
REL_ID = {r: 3 + i for i, r in enumerate(ROLE_RELS)}
REL_CITES = 3 + len(ROLE_RELS)
REL_SAME_FIELD = REL_CITES + 1
REL_SAME_SUBFIELD = REL_CITES + 2
REL_DO = REL_CITES + 3                # world model: an action token -> the node(s) it changes
N_RELS = REL_DO + 1

# masking a claim must also hide what paraphrases it
CHILD_RELS = ("supported_by", "challenged_by", "implies")
# optional mask_policy "siblings": KU types that restate each other
PARAPHRASE_SIBLINGS = {"method": ["procedure"], "procedure": ["method"]}

# ---------------------------------------------------------------- cleaning
_PH = re.compile(
    r"^\s*(n/?a|none|nil|-+|not applicable|none (reported|provided|mentioned|stated|noted|identified)"
    r"|not (reported|mentioned|specified|available|provided|stated|discussed)"
    r"|no (contradicting|conflicting) evidence( (is|was))?( (provided|reported|presented|mentioned))?"
    r"|no links?( (to data or code )?(are |is )?provided)?|no (data|code)[^.]{0,60}"
    r"|no figures or tables are present|unknown)\s*\.?\s*$", re.I)


# "The paper does not present any contradicting evidence." and its ~50 variants:
# a sentence asserting ABSENCE of counter-evidence. ~70% of contradicting_evidence
# strings under 80 chars are only this. Dropped sentence by sentence, so
# "No direct contradictory evidence. However, the model assumes X." keeps X.
_NULL_SENT = re.compile(
    r"(\b(does|do|did)\s+not\s+(explicitly\s+|directly\s+)?(provide|present|mention|report|discuss|include"
    r"|offer|identify|contain|show|find|highlight|address|acknowledge|have)\b.{0,40}"
    r"\b(contradict|conflict|counter|opposing|refut)"
    r"|\b(does|do|did)\s+not\s+(directly\s+)?contradict"
    r"|\bno\s+(explicit\s+|direct\s+|specific\s+|significant\s+|clear\s+)?(contradict\w*|conflict\w*|counter\w*"
    r"|opposing|refuting)\s+(evidence|results?|findings?|data)"
    r"|\bno\s+(direct\s+|experimental\s+)?evidence\s+(is\s+)?(presented\s+|provided\s+)?(that\s+|to\s+)?"
    r"(contradict|refut|against)"
    r"|\bthere\s+is\s+no\s+(direct\s+)?(contradict|conflict|evidence\s+(that\s+|to\s+)?contradict))", re.I)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def strip_null_statements(s):
    keep = [x for x in _SENT.split(s) if not _NULL_SENT.search(x) and not _PH.match(x)]
    return " ".join(keep).strip()


def clean(text, min_chars, null_statements=False):
    """Text, or None when missing / placeholder / null statement / too short.
    Returns (text, why)."""
    if text is None:
        return None, "missing"
    s = " ".join(str(text).split())
    if not s:
        return None, "missing"
    if _PH.match(s):
        return None, "placeholder"
    if null_statements:
        s2 = strip_null_statements(s)
        if len(s2) < min_chars:
            return None, "null_statement"
        s = s2
    if len(s) < min_chars:
        return None, "short"
    return s, "ok"


def parse_claims(raw):
    if not raw:
        return []
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return []
    return [c for c in v if isinstance(c, dict)] if isinstance(v, list) else []


# ---------------------------------------------------------------- metadata
_NEW_ID = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}")
_OLD_ID = re.compile(r"^[a-z\-]+(?:\.[A-Z]{2})?/(\d{2})\d{5}")


def paper_year(rec):
    for k in ("source_year", "oa_year"):
        m = re.match(r"^\s*(\d{4})", str(rec.get(k) or ""))
        if m and 1900 < int(m.group(1)) < 2100:
            return int(m.group(1))
    pid = str(rec.get("paper_id") or "")
    m = _NEW_ID.match(pid) or _OLD_ID.match(pid)
    if m:
        yy = int(m.group(1))
        return 2000 + yy if yy < 90 else 1900 + yy
    return None


def split_field(fs):
    s = " ".join(str(fs or "").split())
    for sep in (" — ", "—", " – ", " - "):
        if sep in s:
            a, b = s.split(sep, 1)
            return a.strip() or "Unknown", s
    return (s or "Unknown"), (s or "Unknown")


PAPER_TYPES = ["theoretical", "methodological", "empirical", "review", "other"]


def paper_type(t):
    """Primary paper type (first one named)."""
    s = str(t or "").lower()
    hits = sorted((s.find(p), p) for p in PAPER_TYPES[:-1] if p in s)
    return PAPER_TYPES.index(hits[0][1]) if hits else PAPER_TYPES.index("other")


def openalex_key(url):
    return str(url).rstrip("/").split("/")[-1].strip() if url else None


def referenced(rec):
    raw = rec.get("oa_referenced_works")
    if not raw:
        return []
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return []
    return [openalex_key(u) for u in v if isinstance(u, str) and u.strip()] if isinstance(v, list) else []


# ---------------------------------------------------------------- KU flags
_NUM = re.compile(r"\d+(\.\d+)?\s*(%|x\b|×|times\b|dB|GeV|K\b|ms\b|s\b|nm\b|eV\b)|\b\d+\.\d+\b")
_FIG = re.compile(r"\b(fig(ure)?s?\.?|tables?|eq(uation)?s?\.?)\s*\(?\d", re.I)


def ku_flags(text):
    return {"quantitative": bool(_NUM.search(text)), "cites_figure": bool(_FIG.search(text))}


# ---------------------------------------------------------------- paper -> one KU
_SECTION_TITLE = {"context": "Research context", "question": "Research question", "method": "Methods",
                  "procedure": "Procedures and architectures", "result": "Key results",
                  "interpretation": "Interpretation", "limitation": "Limitations", "figures": "Figures and tables",
                  "robustness": "Robustness", "data_code": "Data and code", "ethics": "Ethics"}


def paper_ku(kus, edges):
    """ku_mode="paper": collapse a paper's KUs (from record_to_kus, already cleaned)
    into ONE KU holding the full text: every section, then every claim with its
    supporting / contradicting evidence and implications. -> (kus, edges)."""
    parts = [f"{_SECTION_TITLE.get(t, t)}: {txt}" for t, txt, _ in kus if t in _SECTION_TITLE]
    kids = {}
    for a, r, b in edges:
        if r in CHILD_RELS:
            kids.setdefault(a, []).append((r, b))
    label = {"supported_by": "Supporting evidence", "challenged_by": "Contradicting evidence",
             "implies": "Implications"}
    n = 0
    for i, (t, txt, _) in enumerate(kus):
        if t != "claim":
            continue
        n += 1
        claim = [f"Claim {n}: {txt}"] + [f"{label[r]}: {kus[b][1]}" for r, b in kids.get(i, [])]
        parts.append(" ".join(claim))
    text = "\n\n".join(parts)
    flags = ku_flags(text)
    flags["challenged"] = any(r == "challenged_by" for r, _ in sum(kids.values(), []))
    return [(PAPER_KU, text, flags)], []


# ---------------------------------------------------------------- paper_masked: query templates
# ku_mode="paper_masked": the target's own paper is shown as ONE text made of the "given" KU
# types (joined, encoded together), and the model predicts the hidden KU. Templates are the
# reasoning queries (tasks.py) plus "all-<t>": everything except t and the KUs that restate t.
PROBLEM = ("context", "question")
LEAK_FAMILY = {                       # hidden together with the target: they paraphrase it
    "context": ("context",), "question": ("question",),
    "method": ("method", "procedure"), "procedure": ("procedure", "method"),
    "result": ("result", "interpretation", "claim", "evidence", "implication"),
    "interpretation": ("interpretation", "implication"), "limitation": ("limitation",),
    "claim": ("claim", "evidence", "implication"), "evidence": ("evidence", "claim"),
    "implication": ("implication", "claim", "interpretation"),
}
TEMPLATES = {                         # name -> (given KU types, target KU types)
    "forward": (PROBLEM + ("method", "procedure"), ("result",)),
    "inverse": (PROBLEM + ("result",), ("method", "procedure")),
    "abduction": (("result", "procedure"), ("claim", "question")),
}
TASK_TEMPLATES = tuple(TEMPLATES)
for _t in ALL_TARGET_TYPES:
    TEMPLATES[f"all-{_t}"] = (tuple(x for x in ALL_TARGET_TYPES if x not in LEAK_FAMILY[_t]), (_t,))
TEMPLATE_NAMES = list(TEMPLATES)


def active_templates(which):
    """pm_templates: "full" = the 3 reasoning queries + leave-one-out for every target type
    (13 texts per paper); "tasks" = the 3 reasoning queries only (training targets are then
    result, method, procedure, claim, question)."""
    return TEMPLATE_NAMES if which == "full" else list(TASK_TEMPLATES)


def template_for(given):
    """The template whose given set is exactly `given` (None if there is none)."""
    g = set(given)
    for name, (gv, _) in TEMPLATES.items():
        if set(gv) == g:
            return name
    return None


def compose_text(kus, edges, given, replace=None):
    """The visible part of a paper as ONE text: the given sections, then the claims block
    (claim / evidence / implication, each only if given). replace: {type: text} swaps one
    section (counterfactual interventions)."""
    replace = replace or {}
    given = set(given)
    parts = [f"{_SECTION_TITLE.get(t, t)}: {replace.get(t, txt)}" for t, txt, _ in kus
             if t in given and t in _SECTION_TITLE]
    kids = {}
    for a, r, b in edges:
        if r in CHILD_RELS:
            kids.setdefault(a, []).append((r, b))
    label = {"supported_by": "Supporting evidence", "challenged_by": "Contradicting evidence",
             "implies": "Implications"}
    kid_type = {"supported_by": "evidence", "challenged_by": "evidence", "implies": "implication"}
    n = 0
    for i, (t, txt, _) in enumerate(kus):
        if t != "claim":
            continue
        n += 1
        claim = [f"Claim {n}: {txt}"] if "claim" in given else []
        claim += [f"{label[r]}: {kus[b][1]}" for r, b in kids.get(i, []) if kid_type[r] in given]
        if claim:
            parts.append(" ".join(claim))
    return "\n\n".join(parts)


# ---------------------------------------------------------------- minimal-pair counterfactuals
# Evaluation distractors in the BrainBench spirit: the TRUE result / claim with its meaning
# changed by a small edit, so topic and wording stay the same and only reasoning about the
# content separates them. Rule-based (no LLM): direction reversal, negation, changed number.
# A sample must be checked by experts before the numbers are published.
MP_TYPES = ("result", "claim")
_ANT = [("increase", "decrease"), ("increases", "decreases"), ("increased", "decreased"),
        ("increasing", "decreasing"), ("higher", "lower"), ("highest", "lowest"), ("improve", "degrade"),
        ("improves", "degrades"), ("improved", "degraded"), ("improvement", "degradation"),
        ("improvements", "degradations"), ("outperform", "underperform"), ("outperforms", "underperforms"),
        ("outperformed", "underperformed"), ("more", "less"), ("larger", "smaller"), ("largest", "smallest"),
        ("faster", "slower"), ("better", "worse"), ("best", "worst"), ("positive", "negative"),
        ("stronger", "weaker"), ("enhances", "reduces"), ("enhanced", "reduced"), ("gain", "loss"),
        ("gains", "losses"), ("effective", "ineffective"), ("efficient", "inefficient"),
        ("accurate", "inaccurate"), ("stable", "unstable"), ("consistent", "inconsistent"),
        ("supports", "contradicts"), ("confirms", "refutes"), ("success", "failure"), ("above", "below"),
        ("maximum", "minimum"), ("rise", "fall"), ("significant", "non-significant"), ("robust", "fragile")]
_ANT_MAP = {**{a: b for a, b in _ANT}, **{b: a for a, b in _ANT}}
_ANT_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _ANT_MAP), key=len, reverse=True)) + r")\b", re.I)
_NEG = [(r"\bis\b", "is not"), (r"\bare\b", "are not"), (r"\bcan\b", "cannot"), (r"\bshows\b", "does not show"),
        (r"\bdemonstrates\b", "does not demonstrate"), (r"\bachieves\b", "fails to achieve"),
        (r"\bleads to\b", "does not lead to"), (r"\byields\b", "does not yield")]
_NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\s*%?)")


def _case(src, dst):
    return dst.capitalize() if src[:1].isupper() else dst


def counterfactuals(text):
    """-> [(kind, edited text)]: up to three minimal edits that change the meaning."""
    out = []
    flipped = _ANT_RE.sub(lambda m: _case(m.group(0), _ANT_MAP[m.group(0).lower()]), text)
    if flipped != text:
        out.append(("direction", flipped))
    for pat, rep in _NEG:
        neg = re.sub(pat, rep, text, count=1)
        if neg != text:
            out.append(("negation", neg)); break
    m = _NUM.search(text)
    if m:
        v = float(m.group(1)); nv = v * 0.5 if v >= 2 else v * 2 + 1
        s = (f"{nv:.{len(m.group(1).split('.')[1])}f}" if "." in m.group(1) else str(int(round(nv))))
        if s != m.group(1):
            out.append(("number", text[:m.start(1)] + s + text[m.end(1):]))
    return out


# ---------------------------------------------------------------- record -> KUs
def record_to_kus(rec, min_chars=20, optional=()):
    """-> (kus, edges, stats). kus: list of (type, text, flags). edges: list of
    (src index, relation, dst index) within the record."""
    kus, edges, stats = [], [], {}
    first = {}

    def add(t, text, null_statements=False, tag=None):
        txt, why = clean(text, min_chars, null_statements)
        key = (tag or t, why)
        stats[key] = stats.get(key, 0) + 1
        if txt is None:
            return None
        kus.append((t, txt, ku_flags(txt)))
        first.setdefault(t, len(kus) - 1)
        return len(kus) - 1

    for t, col in SECTION_KUS.items():
        add(t, rec.get(col))
    for t, col in OPTIONAL_KUS.items():
        if t in optional:
            add(t, rec.get(col))
    for c in parse_claims(rec.get("claims")):
        ci = add("claim", c.get("description") or c.get("details"))
        if ci is None:
            continue
        challenged = False
        for key, rel in (("supporting_evidence", "supported_by"), ("contradicting_evidence", "challenged_by")):
            ei = add("evidence", c.get(key), null_statements=(rel == "challenged_by"), tag=key)
            if ei is not None:
                edges.append((ci, rel, ei))
                challenged |= rel == "challenged_by"
        ii = add("implication", c.get("implications"))
        if ii is not None:
            edges.append((ci, "implies", ii))
        kus[ci][2]["challenged"] = challenged
    for s, rel, d in SCHEMA_EDGES:
        if rel in CHILD_RELS or s not in first:
            continue
        if s == "result" and d == "claim":
            edges += [(first[s], rel, k) for k, ku in enumerate(kus) if ku[0] == "claim"]
        elif d in first:
            edges.append((first[s], rel, first[d]))
    return kus, edges, stats
