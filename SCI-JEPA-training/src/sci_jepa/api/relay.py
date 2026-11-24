"""
Relay worker for the hosted playground (deploy/playground). The API runs on a cluster GPU that only the TIB network
can reach; this worker lets a public website use it without opening any port. Every connection goes out from here:

    browser --> https://<playground>/api/call --> queue (Upstash Redis, in the Vercel project)
    this worker --(long poll, outbound HTTPS)--> /api/relay/pull --> one job
                --> http://<api>/v1/...          (the API, inside the cluster)
                --> /api/relay/done              --> the browser's waiting /api/call returns the answer

Run it anywhere that reaches both the API and the internet (the GPU node itself, or the login node):

    export PLAYGROUND_RELAY_TOKEN=<RELAY_TOKEN of the Vercel project>
    python -m sci_jepa.api.relay --api http://localhost:8000 --relay https://scijepa-playground.vercel.app

Standard library only. Whatever the queue holds, the worker sends only the routes in ALLOWED, and only to --api.
Each website visitor reaches the API as their own client (`Authorization: Bearer playground-<hash of their address>`),
so the API's per-client rate limits apply per visitor. The API never logs the Authorization header.
"""
import argparse
import json
import os
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

USER_AGENT = "sci-jepa-playground-relay/1"
JOB_ID = re.compile(r"^pg_[0-9a-f]{16}$")
CLIENT = re.compile(r"^[0-9a-f]{8,64}$")
_SES = r"ses_[A-Za-z0-9_-]{1,64}"
# (method, full-match path). Keep in step with deploy/playground/lib/routes.ts.
ALLOWED = [
    ("GET", re.compile(r"/v1/(health|ready|info|models|tasks|papers/search)")),
    ("GET", re.compile(r"/v1/papers/[A-Za-z0-9._-]{1,32}(/[A-Za-z0-9._-]{1,32})?")),   # 2403.00015, hep-th/9901001
    ("GET", re.compile(r"/v1/kus/ku_\d{1,12}")),
    ("GET", re.compile(rf"/v1/sessions/{_SES}")),
    ("DELETE", re.compile(rf"/v1/sessions/{_SES}")),
    ("POST", re.compile(r"/v1/(ask|ask/paper|parse|validate|consistency|edit-check|intervention"
                        r"|predict/(forward|inverse|abduction))")),
    ("POST", re.compile(rf"/v1/sessions/{_SES}/answers")),
]
QUERY_KEYS = {"q", "limit", "cursor"}
MAX_BODY = 64 * 1024            # the API's own --max-body default
MAX_ANSWER = 1024 * 1024


def allowed(method, path):
    if not isinstance(method, str) or not isinstance(path, str) or ".." in path or "//" in path:
        return False
    return any(m == method and rx.fullmatch(path) for m, rx in ALLOWED)


def check_job(job):
    """The job if it is one this worker may forward, else a reason (str)."""
    if not isinstance(job, dict) or not JOB_ID.match(str(job.get("id", ""))):
        return "bad id"
    if not allowed(job.get("method"), job.get("path")):
        return f"route not allowed: {job.get('method')} {job.get('path')}"
    q = job.get("query")
    if q is not None and (not isinstance(q, dict) or set(q) - QUERY_KEYS
                          or any(not isinstance(v, (str, int)) or len(str(v)) > 200 for v in q.values())):
        return "bad query"
    if job.get("body") is not None and len(json.dumps(job["body"])) > MAX_BODY:
        return "body too large"
    if not CLIENT.match(str(job.get("client", ""))):
        return "bad client"
    return job


def _problem(status, title, code, detail=None):
    return {"type": f"https://sci-jepa.org/problems/{code}", "title": title, "status": status, "code": code,
            "detail": detail}


def forward(api, job, timeout=150):
    """Send one job to the API -> the answer envelope {id, status, body, request_id, model_id, ms}."""
    url = api.rstrip("/") + urllib.parse.quote(job["path"], safe="/._-")
    if job.get("query"):
        url += "?" + urllib.parse.urlencode({k: str(v) for k, v in job["query"].items()})
    data = None if job.get("body") is None else json.dumps(job["body"]).encode()
    headers = {"User-Agent": USER_AGENT, "X-Request-Id": job["id"], "Accept": "application/json",
               "Authorization": f"Bearer playground-{job['client']}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=job["method"], headers=headers)
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw, hdrs = r.status, r.read(MAX_ANSWER + 1), r.headers
    except urllib.error.HTTPError as e:
        status, raw, hdrs = e.code, e.read(MAX_ANSWER + 1), e.headers
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"id": job["id"], "status": 502, "ms": round(1000 * (time.monotonic() - t0)), "request_id": None,
                "model_id": None, "body": _problem(502, "The model server did not answer", "upstream_unreachable",
                                                   f"{type(e).__name__}: {e}")}
    if len(raw) > MAX_ANSWER:
        body = _problem(502, "The answer is too large for the playground", "answer_too_large")
        status = 502
    else:
        text = raw.decode("utf-8", "replace")
        try:
            body = json.loads(text) if text else None
        except ValueError:
            body = text
    return {"id": job["id"], "status": status, "body": body, "ms": round(1000 * (time.monotonic() - t0)),
            "request_id": hdrs.get("X-Request-Id"), "model_id": hdrs.get("X-Model-Id"),
            "retry_after": hdrs.get("Retry-After")}


class Relay:
    def __init__(self, api, relay, token, workers=4, wait=25, log=None):
        self.api, self.relay, self.token = api.rstrip("/"), relay.rstrip("/"), token
        self.workers, self.wait = workers, wait
        self.log = log or (lambda *a: print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True))
        self.ready, self._ready_at = False, 0.0
        self.stop = threading.Event()
        self.handled = 0

    # -- the website
    def _req(self, method, path, body=None, timeout=30):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.relay + path, data=data, method=method, headers={
            "Authorization": f"Bearer {self.token}", "User-Agent": USER_AGENT,
            **({"Content-Type": "application/json"} if data is not None else {})})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()

    def pull(self):
        """Wait up to `wait` s for one job (the website holds the request open). None when there was none."""
        q = urllib.parse.urlencode({"wait": self.wait, "ready": int(self.ready), "workers": self.workers})
        status, raw = self._req("GET", f"/api/relay/pull?{q}", timeout=self.wait + 20)
        return json.loads(raw) if status == 200 and raw else None

    def done(self, answer):
        for i in range(3):
            try:
                self._req("POST", "/api/relay/done", answer)
                return
            except Exception as e:                       # noqa: BLE001 - keep serving whatever went wrong
                self.log(f"posting answer {answer['id']} failed ({e}); retry {i + 1}/3")
                time.sleep(1 + 2 * i)

    # -- the API
    def check_ready(self):
        if time.monotonic() - self._ready_at < 30:
            return self.ready
        self._ready_at = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(self.api + "/v1/ready",
                                                               headers={"User-Agent": USER_AGENT}), timeout=5) as r:
                ok = r.status == 200
        except Exception:                                # noqa: BLE001 - 503 while loading, or down
            ok = False
        if ok != self.ready:
            self.log(f"API {self.api} is {'ready' if ok else 'not ready'}")
        self.ready = ok
        return ok

    def handle(self, job):
        why = check_job(job)
        if isinstance(why, str):
            self.log(f"refused job {job.get('id') if isinstance(job, dict) else '?'}: {why}")
            if isinstance(job, dict) and JOB_ID.match(str(job.get("id", ""))):
                self.done({"id": job["id"], "status": 400, "ms": 0, "request_id": None, "model_id": None,
                           "body": _problem(400, "The playground cannot send this request", "not_allowed", why)})
            return
        ans = forward(self.api, job)
        self.done(ans)
        self.handled += 1
        self.log(f"{job['method']} {job['path']} -> {ans['status']} in {ans['ms']} ms ({job['id']})")

    def serve(self):
        self.log(f"relay {self.relay} <-> API {self.api}, {self.workers} workers")
        slots = threading.Semaphore(self.workers)
        backoff = 1.0
        with ThreadPoolExecutor(self.workers, thread_name_prefix="relay") as pool:
            while not self.stop.is_set():
                slots.acquire()                          # pull only when a worker is free
                self.check_ready()
                try:
                    job = self.pull()
                    backoff = 1.0
                except Exception as e:                   # noqa: BLE001 - network trouble: back off and retry
                    slots.release()
                    code = getattr(e, "code", None)
                    if code in (401, 403):
                        self.log("the website refused the relay token (PLAYGROUND_RELAY_TOKEN); stopping")
                        return 2
                    self.log(f"pull failed ({e}); retrying in {backoff:.0f} s")
                    self.stop.wait(backoff)
                    backoff = min(60.0, backoff * 2)
                    continue
                if job is None:
                    slots.release()
                    continue

                def run(j=job):
                    try:
                        self.handle(j)
                    except Exception as e:               # noqa: BLE001
                        self.log(f"job failed: {e}")
                    finally:
                        slots.release()
                pool.submit(run)
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m sci_jepa.api.relay", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default=os.environ.get("SCIJEPA_API", "http://localhost:8000"),
                    help="the API inside the cluster (env SCIJEPA_API; default http://localhost:8000)")
    ap.add_argument("--relay", default=os.environ.get("PLAYGROUND_RELAY_URL"),
                    help="the playground website (env PLAYGROUND_RELAY_URL)")
    ap.add_argument("--workers", type=int, default=4, help="requests forwarded at once (the GPU queues the rest)")
    ap.add_argument("--wait", type=int, default=25, help="seconds one long poll waits for a job (max 25)")
    a = ap.parse_args(argv)
    token = os.environ.get("PLAYGROUND_RELAY_TOKEN")     # not a flag: command lines are visible in `ps`
    if not a.relay or not token:
        ap.error("set --relay (or PLAYGROUND_RELAY_URL) and the environment variable PLAYGROUND_RELAY_TOKEN")
    r = Relay(a.api, a.relay, token, workers=max(1, a.workers), wait=max(1, min(25, a.wait)))
    signal.signal(signal.SIGTERM, lambda *_: r.stop.set())
    try:
        return r.serve()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
