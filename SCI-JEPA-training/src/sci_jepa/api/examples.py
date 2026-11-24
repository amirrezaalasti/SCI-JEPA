"""
Examples for the interactive docs (/v1/docs): one request per endpoint that the server accepts, and the
response the deployed model gave to it.

REQUESTS holds the inputs, written by hand. examples_responses.json holds the outputs, recorded from a
running server by scripts/check_api_examples.py:

    python scripts/check_api_examples.py --url http://<node>:8000            # send every example, compare shapes
    python scripts/check_api_examples.py --url http://<node>:8000 --write    # and record the responses again

Path and query parameters of the GET endpoints are in PARAMS. Sessions are created by the `need_more`
example of POST /v1/ask, so the session endpoints have no fixed ID.
"""
import json
from pathlib import Path

FORWARD_PARTS = {
    "problem": "Image classifiers overfit on small medical datasets.",
    "method": "Fine-tuning a pretrained vision transformer with strong data augmentation and dropout.",
    "conditions": "About 5,000 labelled chest X-rays, 80/20 train/test split, five random seeds.",
}
ASK_TEXT = ("We want to predict reaction yields from written lab protocols. "
            "We fine-tune a small language model on the protocol text.")

# endpoint -> {example name: (summary, request body)}. The first example is the one "Try it out" fills in.
REQUESTS = {
    "ask": {
        "complete": ("Problem, method and conditions: answers at once", {
            "text": ASK_TEXT + " We use about 50,000 Suzuki coupling protocols from US patents with an 80/20 split.",
            "goal": "forward", "options": {"top_k": 2}}),
        "need_more": ("Conditions missing: the server asks for them", {
            "text": ASK_TEXT, "goal": "forward", "options": {"top_k": 2}}),
    },
    "session_answers": {
        "conditions": ("Answer the missing part of a need_more session", {
            "answers": {"conditions": "About 50,000 Suzuki coupling protocols from US patents, 80/20 split."}}),
    },
    "parse": {
        "forward": ("Split free text into parts", {
            "text": ASK_TEXT + " We use about 50,000 Suzuki coupling protocols from US patents with an 80/20 split.",
            "goal": "forward"}),
    },
    "validate": {
        "valid": ("All parts the task needs", {"task": "forward", "parts": FORWARD_PARTS}),
        "missing_part": ("A part is missing", {"task": "forward", "parts": {
            "problem": FORWARD_PARTS["problem"], "method": FORWARD_PARTS["method"]}}),
    },
    "forward": {
        "forward": ("Problem, method, conditions -> results", {"parts": FORWARD_PARTS, "options": {"top_k": 3}}),
        "filtered": ("Only Computer Science papers from 2020 to 2025", {"parts": FORWARD_PARTS, "options": {
            "top_k": 3, "scoring": "hybrid",
            "filters": {"fields": ["Computer Science"], "year_from": 2020, "year_to": 2025}}}),
    },
    "inverse": {
        "inverse": ("Problem, result -> methods", {"parts": {
            "problem": FORWARD_PARTS["problem"],
            "result": "Top-1 accuracy rose from 71.2% to 76.8% over a ResNet-50 baseline on the held-out test set."},
            "options": {"top_k": 3}}),
    },
    "abduction": {
        "abduction": ("Result, conditions -> explanations", {"parts": {
            "result": "Galaxy rotation curves stay flat far beyond the visible disk, at about 200 km/s out to 30 kpc.",
            "conditions": "HI 21 cm observations of 40 nearby spiral galaxies with the Very Large Array."},
            "options": {"top_k": 3}}),
    },
    "consistency": {
        "consistency": ("Claim and evidence", {"parts": {
            "claim": "Dropout reduces overfitting in small transformers.",
            "evidence": "On 5 datasets, validation loss fell by 12% with dropout 0.1 versus none."}, "related": 2}),
    },
    "edit_check": {
        "direction": ("A finding and a copy with the direction flipped", {
            "original": "Adding dropout reduced overfitting by 12%.",
            "edited": "Adding dropout increased overfitting by 12%."}),
    },
    "intervention": {
        "method": ("Swap the method", {"parts": {
            "problem": "Predicting reaction yields from written lab protocols.",
            "method": "Fine-tuning a small language model on the protocol text.",
            "conditions": "About 50,000 Suzuki coupling protocols from US patents, 80/20 split."},
            "change": {"part": "method", "text": "Training a graph neural network on reaction graphs."},
            "options": {"top_k": 2}}),
    },
    "ask_paper": {
        "forward": ("A validation paper, with the true answer shown", {
            "paper_id": "2403.00015", "goal": "forward", "reveal_truth": True, "options": {"top_k": 2}}),
    },
}

# Error answers (RFC 9457), recorded per endpoint so each one shows an error that endpoint really gives.
# endpoint -> {status: (summary, request)}; request holds "body", "path" and "query" as the call needs them.
# A "session_id" of None means: open a fresh session first (an error that needs a live session).
_VAGUE_RESULT = "it worked well"
ERRORS = {
    "ask": {
        "400": ("No `text`", {"body": {"goal": "forward"}}),
        "422": ("A model this server does not serve (Swagger's placeholder \"string\")", {"body": {
            "text": ASK_TEXT, "goal": "forward", "model": "string"}}),
    },
    "parse": {
        "400": ("No `text`", {"body": {"goal": "forward"}}),
        "422": ("An unknown goal", {"body": {"text": ASK_TEXT, "goal": "sideways"}}),
    },
    "validate": {
        "400": ("No `task`", {"body": {"parts": FORWARD_PARTS}}),
        "422": ("An unknown task", {"body": {"task": "sideways", "parts": FORWARD_PARTS}}),
    },
    "forward": {
        "400": ("Parts sent at the top level instead of inside `parts`", {"body": dict(FORWARD_PARTS)}),
        "422": ("A part too vague to search with", {"body": {
            "parts": {**FORWARD_PARTS, "method": "we did experiments on the data"}}}),
    },
    "inverse": {
        "400": ("Parts sent at the top level instead of inside `parts`", {"body": {
            "problem": FORWARD_PARTS["problem"], "result": "Top-1 accuracy rose from 71.2% to 76.8%."}}),
        "422": ("A result too vague to search with", {"body": {"parts": {
            "problem": FORWARD_PARTS["problem"], "result": _VAGUE_RESULT}}}),
    },
    "abduction": {
        "400": ("Parts sent at the top level instead of inside `parts`", {"body": {
            "result": "Galaxy rotation curves stay flat far beyond the visible disk.",
            "conditions": FORWARD_PARTS["conditions"]}}),
        "422": ("A result too vague to search with", {"body": {"parts": {
            "result": _VAGUE_RESULT, "conditions": FORWARD_PARTS["conditions"]}}}),
    },
    "consistency": {
        "400": ("Parts sent at the top level instead of inside `parts`", {"body": {
            "claim": "Dropout reduces overfitting in small transformers.",
            "evidence": "On 5 datasets, validation loss fell by 12% with dropout 0.1 versus none."}}),
        "422": ("`related` out of range", {"body": {"parts": {
            "claim": "Dropout reduces overfitting in small transformers.",
            "evidence": "On 5 datasets, validation loss fell by 12% with dropout 0.1 versus none."}, "related": 1000}}),
    },
    "edit_check": {
        "400": ("No `edited`", {"body": {"original": "Adding dropout reduced overfitting by 12%."}}),
        "422": ("The edited finding is the same as the original", {"body": {
            "original": "Adding dropout reduced overfitting by 12%.",
            "edited": "Adding dropout reduced overfitting by 12%."}}),
    },
    "intervention": {
        "400": ("No `change`", {"body": {"parts": FORWARD_PARTS}}),
        "422": ("A change to a part that does not exist", {"body": {"parts": FORWARD_PARTS, "change": {
            "part": "colour", "text": "Training a graph neural network on reaction graphs."}}}),
    },
    "session_answers": {
        "400": ("Answers sent at the top level instead of inside `answers`", {
            "path": {"session_id": None}, "body": {"conditions": "About 50,000 Suzuki coupling protocols."}}),
        "404": ("A session that was deleted or expired", {
            "path": {"session_id": "ses_doesnotexist"},
            "body": {"answers": {"conditions": "About 50,000 Suzuki coupling protocols from US patents."}}}),
        "422": ("An answer for a part that does not exist", {"path": {"session_id": None}, "body": {
            "answers": {"colour": "About 50,000 Suzuki coupling protocols from US patents, 80/20 split."}}}),
    },
    "session_get": {"404": ("A session that was deleted or expired", {"path": {"session_id": "ses_doesnotexist"}})},
    "session_delete": {"404": ("A session that was deleted or expired", {"path": {"session_id": "ses_doesnotexist"}})},
    "papers_search": {"422": ("An empty query", {"query": {"q": ""}})},
    "paper": {
        "404": ("A paper that is not in the corpus", {"path": {"paper_id": "9999.99999"}}),
        "422": ("Neither an arXiv ID nor a DOI", {"path": {"paper_id": "not-a-paper"}}),
    },
    "ku": {"404": ("A KU that does not exist", {"path": {"ku_id": "ku_999999999"}})},
    "ask_paper": {
        "400": ("No `paper_id`", {"body": {"goal": "forward"}}),
        "403": ("reveal_truth for a training paper", {"body": {
            "paper_id": "2210.12843", "goal": "forward", "reveal_truth": True}}),
        "404": ("A paper that is not in the corpus", {"body": {"paper_id": "9999.99999", "goal": "forward"}}),
        "422": ("An unknown goal", {"body": {"paper_id": "2403.00015", "goal": "sideways"}}),
    },
}

# GET endpoints: path or query parameters of the documented call.
PARAMS = {
    "health": {}, "ready": {}, "info": {}, "models": {}, "tasks": {},
    "papers_search": {"q": "graph neural networks for chemical reaction yield prediction", "limit": 3},
    "paper": {"paper_id": "2403.00015"},
    "ku": {"ku_id": "ku_17441711"},
}

# endpoint -> (method, path template)
ROUTES = {
    "health": ("GET", "/v1/health"), "ready": ("GET", "/v1/ready"), "info": ("GET", "/v1/info"),
    "models": ("GET", "/v1/models"), "tasks": ("GET", "/v1/tasks"),
    "ask": ("POST", "/v1/ask"), "session_answers": ("POST", "/v1/sessions/{session_id}/answers"),
    "session_get": ("GET", "/v1/sessions/{session_id}"), "session_delete": ("DELETE", "/v1/sessions/{session_id}"),
    "parse": ("POST", "/v1/parse"), "validate": ("POST", "/v1/validate"),
    "forward": ("POST", "/v1/predict/forward"), "inverse": ("POST", "/v1/predict/inverse"),
    "abduction": ("POST", "/v1/predict/abduction"), "consistency": ("POST", "/v1/consistency"),
    "edit_check": ("POST", "/v1/edit-check"), "intervention": ("POST", "/v1/intervention"),
    "papers_search": ("GET", "/v1/papers/search"), "paper": ("GET", "/v1/papers/{paper_id}"),
    "ku": ("GET", "/v1/kus/{ku_id}"), "ask_paper": ("POST", "/v1/ask/paper"),
}

RESPONSES_FILE = Path(__file__).with_name("examples_responses.json")


def load_responses():
    """{endpoint: {example name: response body}} as recorded; {} when the file is missing."""
    try:
        return json.loads(RESPONSES_FILE.read_text())
    except (OSError, ValueError):
        return {}


def body(endpoint):
    """Body(openapi_examples=...) for a POST endpoint."""
    return {name: {"summary": summary, "value": value} for name, (summary, value) in REQUESTS[endpoint].items()}


def param(endpoint, name):
    """openapi_examples for one path or query parameter of a GET endpoint."""
    return {"example": {"value": PARAMS[endpoint][name]}}


def responses(endpoint, errors=None):
    """`responses=` for a route: the recorded 200 answers, and the RFC 9457 errors."""
    errors = ("400", "422") if errors is None else errors
    rec = load_responses()
    ok = rec.get(endpoint) or {}
    out = {}
    if ok:
        content = {"examples": {name: {"summary": REQUESTS.get(endpoint, {}).get(name, ("Example answer",))[0],
                                       "value": v}
                                for name, v in ok.items()}}
        out["200"] = {"description": "Successful answer", "content": {"application/json": content}}
    for code in errors:
        e = rec.get("_errors", {}).get(endpoint, {}).get(code)
        content = {"example": e} if e else {}
        if e and code in ERRORS.get(endpoint, {}):
            content = {"examples": {code: {"summary": ERRORS[endpoint][code][0], "value": e}}}
        out[code] = {"description": ERROR_TEXT[code], "content": {"application/problem+json": content}}
    return out


def keep_nulls(spec):
    """FastAPI drops null values from examples when it builds the OpenAPI file, so documented answers would lack
    keys such as `title` or `arxiv_id`. Put the examples back exactly as written and recorded (in place)."""
    rec = load_responses()
    name_of = {(m.lower(), p): n for n, (m, p) in ROUTES.items()}
    for path, ops in spec.get("paths", {}).items():
        for method, op in ops.items():
            name = name_of.get((method, path))
            if name is None:
                continue
            ex = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("examples", {})
            for k, v in ex.items():
                if k in REQUESTS.get(name, {}):
                    v["value"] = REQUESTS[name][k][1]
            for code, r in op.get("responses", {}).items():
                c = r.get("content", {})
                for k, v in c.get("application/json", {}).get("examples", {}).items():
                    if k in rec.get(name, {}):
                        v["value"] = rec[name][k]
                err = rec.get("_errors", {}).get(name, {}).get(code)
                for v in c.get("application/problem+json", {}).get("examples", {}).values():
                    if err is not None:
                        v["value"] = err
    return spec


ERROR_TEXT = {"400": "Malformed request: unknown or missing field, or a wrong type",
              "403": "Not allowed, e.g. reveal_truth for a paper outside the validation split",
              "404": "Not found or expired",
              "422": "Valid JSON, but a value is out of range or a part fails the content checks"}
