"""The researchers' playground: the page the API serves at /v1/playground, and the relay worker
(sci_jepa/api/relay.py) that lets the hosted copy (deploy/playground) reach an API inside the cluster."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from sci_jepa.api import relay as R

JOB = {"id": "pg_0123456789abcdef", "method": "POST", "path": "/v1/predict/forward", "query": None,
       "body": {"parts": {"problem": "x"}}, "client": "a1b2c3d4e5f60718", "ts": 0}


# ------------------------------------------------------------------ which requests the worker forwards
@pytest.mark.parametrize("method,path", [
    ("GET", "/v1/info"), ("GET", "/v1/papers/search"), ("GET", "/v1/papers/2403.00015"),
    ("GET", "/v1/papers/hep-th/9901001"), ("GET", "/v1/kus/ku_17441711"), ("GET", "/v1/sessions/ses_ab-C_1"),
    ("DELETE", "/v1/sessions/ses_abc"), ("POST", "/v1/ask"), ("POST", "/v1/ask/paper"),
    ("POST", "/v1/predict/abduction"), ("POST", "/v1/edit-check"), ("POST", "/v1/sessions/ses_abc/answers"),
])
def test_allowed_routes(method, path):
    assert R.allowed(method, path)


@pytest.mark.parametrize("method,path", [
    ("GET", "/metrics"), ("GET", "/v1/docs"), ("POST", "/v1/info"), ("DELETE", "/v1/ask"),
    ("GET", "/v1/papers/../../etc/passwd"), ("GET", "/v1/papers/a/b/c"), ("GET", "//evil.example/v1/info"),
    ("GET", "/v1/info?x=1"), ("POST", "/v1/predict/sideways"), ("GET", "http://evil.example/v1/info"), (None, "/v1/ask"),
])
def test_refused_routes(method, path):
    assert not R.allowed(method, path)


def test_check_job():
    assert R.check_job(dict(JOB)) == JOB
    assert R.check_job({**JOB, "id": "pg_x"}) == "bad id"
    assert R.check_job({**JOB, "query": {"evil": "1"}}) == "bad query"
    assert R.check_job({**JOB, "client": "not hex!"}) == "bad client"
    assert R.check_job({**JOB, "body": {"t": "x" * (R.MAX_BODY + 1)}}) == "body too large"
    assert R.check_job({**JOB, "method": "GET", "path": "/metrics"}).startswith("route not allowed")


# ------------------------------------------------------------------ forwarding to a stand-in API
class FakeAPI(BaseHTTPRequestHandler):
    seen = []

    def _answer(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n)) if n else None
        FakeAPI.seen.append({"method": self.command, "path": self.path, "headers": dict(self.headers), "body": body})
        if self.path.startswith("/v1/papers/9999"):
            out, status = {"title": "Not found", "status": 404, "code": "not_found"}, 404
        elif self.path == "/v1/ready":
            out, status = {"status": "ready"}, 200
        else:
            out, status = {"status": "ok", "echo": body}, 200
        raw = json.dumps(out).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Request-Id", self.headers.get("X-Request-Id", "req_x"))
        self.send_header("X-Model-Id", "sci-jepa-test")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = do_POST = do_DELETE = _answer

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def api():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_forward_post(api):
    FakeAPI.seen.clear()
    a = R.forward(api, JOB)
    assert a["status"] == 200 and a["body"]["echo"] == JOB["body"]
    assert a["request_id"] == JOB["id"] and a["model_id"] == "sci-jepa-test"
    h = FakeAPI.seen[0]["headers"]
    assert h["Authorization"] == f"Bearer playground-{JOB['client']}"    # per-visitor rate limits in the API
    assert h["User-Agent"] == R.USER_AGENT


def test_forward_query_and_errors(api):
    FakeAPI.seen.clear()
    a = R.forward(api, {**JOB, "method": "GET", "path": "/v1/papers/search", "body": None,
                        "query": {"q": "graph nets & yields", "limit": 3}})
    assert a["status"] == 200
    assert FakeAPI.seen[0]["path"] == "/v1/papers/search?q=graph+nets+%26+yields&limit=3"
    a = R.forward(api, {**JOB, "method": "GET", "path": "/v1/papers/9999.99999", "body": None})
    assert a["status"] == 404 and a["body"]["code"] == "not_found"


def test_forward_unreachable():
    a = R.forward("http://127.0.0.1:9", JOB, timeout=2)
    assert a["status"] == 502 and a["body"]["code"] == "upstream_unreachable"


def test_handle_posts_answers(api, monkeypatch):
    r = R.Relay(api, "https://playground.invalid", "tok", log=lambda *a: None)
    sent = []
    monkeypatch.setattr(r, "done", sent.append)
    r.handle(dict(JOB))
    r.handle({**JOB, "id": "pg_ffffffffffffffff", "method": "GET", "path": "/metrics"})
    r.handle({"id": "nonsense"})                         # no valid id: nothing to answer
    assert [s["status"] for s in sent] == [200, 400]
    assert sent[1]["body"]["code"] == "not_allowed" and sent[1]["id"] == "pg_ffffffffffffffff"
    assert r.check_ready() is True


def test_cli_needs_token(monkeypatch):
    monkeypatch.delenv("PLAYGROUND_RELAY_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        R.main(["--relay", "https://playground.invalid"])


# ------------------------------------------------------------------ the page itself
def test_page_served_by_api():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from sci_jepa.api.app import PLAYGROUND, create_app
    eng = SimpleNamespace(lock=threading.Lock(), model_id="m", index_spec="none",
                          state={"model_loaded": False, "index_loaded": False, "search_loaded": False, "error": None})
    with TestClient(create_app(eng, load_in_background=False)) as c:
        r = c.get("/v1/playground")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        assert "<!-- playground-config -->" in r.text        # deploy/playground/scripts/assemble.mjs needs it
        assert c.get("/", follow_redirects=False).headers["location"] == "/v1/playground"
        assert "/v1/playground" not in c.get("/v1/openapi.json").json()["paths"]
    assert PLAYGROUND.exists()
