"""The API call log (sci_jepa/api/calllog.py): every call with its request and answer, the dashboard files, redaction."""
import glob
import json
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from sci_jepa.api import textparts as T  # noqa: E402
from sci_jepa.api.calllog import CallLog  # noqa: E402

TEXT = ("We want to predict reaction yields from lab protocols. "
        "We fine-tune a small language model on the protocol text.")


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from sci_jepa.api.app import create_app
    from sci_jepa.api.engine import Engine
    from sci_jepa.config import PaperAConfig
    from sci_jepa.models.model import PaperAModel
    from sci_jepa.training.train import save_model
    d = tmp_path_factory.mktemp("apilog")
    cfg = PaperAConfig().override(synthetic=True, embedder="hash", device="cpu", index_dir=str(d / "idx"))
    save_model(PaperAModel(cfg), cfg, str(d / "model"))
    eng = Engine(str(d / "model"), device="cpu", index="none", warm_up=False, log=lambda *a: None).load()
    calls = CallLog(str(d / "dash"), host="testhost", port=8000, scrub=T.scrub_pii, flush_every=3600)
    app = create_app(eng, load_in_background=False, call_log=calls)

    def boom():
        raise RuntimeError("kaputt")
    app.add_api_route("/v1/boom", boom)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c, calls, d / "dash"


def test_records_request_and_answer(setup):
    c, calls, _ = setup
    r = c.post("/v1/parse", json={"text": TEXT, "goal": "forward"}, headers={"Authorization": "Bearer secret-token"})
    assert r.status_code == 200
    e = calls.latest(1)[0]
    assert e["id"] == r.headers["X-Request-Id"] and e["method"] == "POST" and e["route"] == "/v1/parse"
    assert e["endpoint"] == "parse" and e["status"] == 200 and e["ms"] >= 0
    assert e["request_body"] == {"text": TEXT, "goal": "forward"}
    assert e["response_body"] == r.json()
    assert e["request_headers"]["authorization"] == "[present, not logged]"
    assert "secret-token" not in json.dumps(e)
    assert e["client"].startswith("key:") and e["summary"]["goal"] == "forward"


def test_errors_and_params(setup):
    c, calls, _ = setup
    r = c.get("/v1/papers/search", params={"q": "x" * 300})
    e = calls.latest(1)[0]
    assert e["status"] == r.status_code >= 400 and e["query"] == {"q": "x" * 300}
    assert e["error"]["code"] == r.json()["code"]
    c.get("/v1/kus/ku_123")
    e = calls.latest(1)[0]
    assert e["route"] == "/v1/kus/{ku_id}" and e["path_params"] == {"ku_id": "ku_123"}


def test_traceback_of_500(setup):
    c, calls, _ = setup
    assert c.get("/v1/boom").status_code == 500
    e = calls.latest(1)[0]
    assert e["status"] == 500 and "RuntimeError: kaputt" in e["traceback"]


def test_probes_kept_apart(setup):
    c, calls, _ = setup
    n = len(calls.latest(1000))
    c.get("/v1/health")
    assert len(calls.latest(1000)) == n
    assert calls.latest(1, probes=True)[0]["path"] == "/v1/health"


def test_files_and_reload(setup):
    c, calls, root = setup
    c.post("/v1/parse", json={"text": TEXT})
    calls._flush()
    snap = json.load(open(root / "apicalls" / "servers" / "testhost-8000.json"))
    assert snap["calls"][0]["path"] == "/v1/parse" and snap["totals"]["calls"] >= 1
    assert snap["totals"]["by_route"]["POST /v1/parse"]["n"] >= 1 and snap["minutes"]
    assert [s["key"] for s in json.load(open(root / "apicalls" / "servers.json"))] == ["testhost-8000"]
    lines = open(glob.glob(str(root / "apicalls" / "log" / "testhost-8000" / "*.jsonl"))[0]).read().splitlines()
    assert len(lines) == calls.total
    again = CallLog(str(root), host="testhost", port=8000, flush_every=3600)   # a restarted server
    assert again.latest(1)[0]["id"] == calls.latest(1)[0]["id"]
    again.close()


def test_redaction(tmp_path):
    body = {"text": "Mail me at a.b@example.org or +49 551 39 12345 about yields.", "n": [1, {"x": "c@d.de"}]}
    pii = CallLog(scrub=T.scrub_pii, redact="pii")
    out = pii.prepare(body)
    assert "example.org" not in json.dumps(out) and "c@d.de" not in json.dumps(out) and out["n"][0] == 1
    meta = CallLog(redact="meta")
    assert meta.prepare(body).startswith("[not logged")
    assert meta.prepare_summary({"text": "secret", "status": "ok"}) == {"status": "ok"}
    with pytest.raises(ValueError):
        CallLog(redact="nope")


def test_snapshot_trims_large_bodies(tmp_path):
    log = CallLog(str(tmp_path), host="h", port=1, snapshot_body_limit=100, flush_every=3600)
    log.record({"id": "r1", "ts": 1.0e9, "method": "POST", "path": "/v1/ask", "route": "/v1/ask", "status": 200,
                "ms": 1.0, "request_body": {"text": "y" * 1000}, "response_body": None})
    s = log.snapshot()["calls"][0]
    assert s["request_body"]["_truncated"] and len(s["request_body"]["preview"]) == 100
    assert os.path.getsize(glob.glob(str(tmp_path / "apicalls" / "log" / "h-1" / "*.jsonl"))[0]) > 1000
    log.close()
