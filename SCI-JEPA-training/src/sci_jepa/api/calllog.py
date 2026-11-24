"""
Call log of the Sci-JEPA API: every request and its answer, in detail, for the dashboard (apicalls.html).

What one call records: request ID, time, method, path, route, path and query parameters, client (address or
hashed bearer token), user agent, request headers (Authorization and cookies are never kept), the request body,
the status, the response headers, the response body, the duration, the error code of an RFC 9457 answer, the
Python traceback of a 500, and a short summary (task, session, status, number of answers).

Where it goes, with a dashboard folder (`sci-jepa api --dashboard <folder>`):

    <folder>/apicalls/servers/<host>-<port>.json   latest calls + totals of one server (the page polls it; pushed
                                                   to the hosted dashboard like every other .json file)
    <folder>/apicalls/servers.json                 index of the servers that wrote to this folder
    <folder>/apicalls/log/<host>-<port>/<YYYY-MM-DD>.jsonl
                                                   every call, one JSON line each (full history; not pushed;
                                                   files older than `days` are deleted)

The snapshot keeps the latest `keep` calls; health, ready, metrics and docs requests ("probes") keep their own
short ring so a load balancer polling /v1/ready does not push real calls out. After a restart the snapshot is
refilled from the day's JSONL file. Without a dashboard folder the log lives in memory only.

Privacy: the request text is what users typed. redact="full" keeps it as sent, "pii" removes emails and phone
numbers (textparts.scrub_pii) from every string, "meta" keeps no bodies at all. The JSONL files keep calls of
deleted sessions; delete them by hand if a user asks.
"""
import collections
import json
import os
import re
import socket
import threading
import time
from datetime import datetime, timezone

PROBES = ("/v1/health", "/v1/ready", "/metrics", "/v1/docs", "/v1/openapi.json", "/favicon.ico",
          "/v1/playground")
DROP_HEADERS = {"authorization", "proxy-authorization", "cookie", "set-cookie"}
REDACT_MODES = ("full", "pii", "meta")


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))[:80]


def _iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _atomic_dump(obj, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=str, separators=(",", ":"))
    os.replace(tmp, path)


def decode_body(raw, content_type, limit):
    """bytes -> (JSON value or text, truncated?). Over `limit` bytes the text is cut (and kept as text)."""
    if not raw:
        return None, False
    cut = len(raw) > limit
    data = raw[:limit] if cut else raw
    text = data.decode("utf-8", "replace")
    if not cut and "json" in (content_type or ""):
        try:
            return json.loads(text), False
        except ValueError:
            pass
    return text + (f"... [{len(raw) - limit} more bytes]" if cut else ""), cut


def _trim(v, limit):
    """A copy of a logged body that fits in about `limit` characters of JSON (for the snapshot)."""
    try:
        s = json.dumps(v, default=str)
    except (TypeError, ValueError):
        s = str(v)
    if len(s) <= limit:
        return v
    return {"_truncated": True, "_bytes": len(s), "preview": s[:limit]}


def _scrub(v, fn):
    if isinstance(v, str):
        return fn(v)[0]
    if isinstance(v, dict):
        return {k: _scrub(x, fn) for k, x in v.items()}
    if isinstance(v, list):
        return [_scrub(x, fn) for x in v]
    return v


def summarise(path, status, req, resp):
    """A few fields the call table shows without opening the call."""
    s = {}
    if isinstance(req, dict):
        for k in ("goal", "task", "paper_id"):
            if req.get(k):
                s[k] = req[k]
        if isinstance(req.get("text"), str):
            s["text"] = req["text"][:160]
    if isinstance(resp, dict):
        for k in ("status", "task", "session", "code", "title", "mode", "valid"):
            if resp.get(k) is not None and k not in s and not (k == "status" and isinstance(resp[k], int)):
                s[k] = resp[k]
        for k in ("answers", "items", "missing", "warnings"):
            if isinstance(resp.get(k), list):
                s[f"n_{k}"] = len(resp[k])
        if isinstance(resp.get("answers"), list) and resp["answers"]:
            a = resp["answers"][0]
            if isinstance(a, dict):
                s["top"] = {k: a[k] for k in ("type", "score", "paper_id", "text") if k in a}
                if isinstance(s["top"].get("text"), str):
                    s["top"]["text"] = s["top"]["text"][:160]
    m = re.match(r"^/v1/sessions/([^/]+)", path)
    if m and "session" not in s:
        s["session"] = m.group(1)
    if status >= 400 and isinstance(resp, dict) and resp.get("errors"):
        s["error_fields"] = [e.get("field") for e in resp["errors"] if isinstance(e, dict)][:5]
    return s


class CallLog:
    def __init__(self, root=None, host=None, port=None, keep=300, keep_probes=30, body_limit=256 * 1024,
                 snapshot_body_limit=24 * 1024, redact="full", days=30, flush_every=1.0, scrub=None, server=None,
                 log=None):
        if redact not in REDACT_MODES:
            raise ValueError(f"redact must be one of {REDACT_MODES}")
        self.root, self.redact, self.days = root, redact, days
        self.body_limit, self.snap_limit, self.flush_every = body_limit, snapshot_body_limit, flush_every
        self.scrub, self.log = scrub, log or (lambda *a: None)
        self.key = _safe(f"{host or socket.gethostname()}-{port or 0}")
        self.calls, self.probes = collections.deque(maxlen=keep), collections.deque(maxlen=keep_probes)
        self.by_route = collections.defaultdict(lambda: {"n": 0, "errors": 0, "ms_sum": 0.0, "ms_max": 0.0})
        self.by_status = collections.Counter()
        self.minutes = collections.OrderedDict()         # minute (epoch s) -> {n, errors, ms_sum}
        self.total = self.errors = 0
        self.started = time.time()
        self.server = {"key": self.key, "host": host or socket.gethostname(), "port": port, "pid": os.getpid(),
                       "started": _iso(self.started), "started_ts": self.started, "redact": redact,
                       **(server or {})}
        self._lock, self._dirty, self._stop = threading.Lock(), False, threading.Event()
        self._thread = None
        if root:
            self.dir = os.path.join(root, "apicalls")
            self.log_dir = os.path.join(self.dir, "log", self.key)
            os.makedirs(os.path.join(self.dir, "servers"), exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            self._prune()
            self._reload()
            self._register()
            self._flush()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="sci-jepa-calllog")
            self._thread.start()

    # ---------------------------------------------------------------- recording
    def prepare(self, body):
        if self.redact == "meta":
            return None if body is None else "[not logged: --call-log-redact meta]"
        if self.redact == "pii" and self.scrub:
            return _scrub(body, self.scrub)
        return body

    def prepare_summary(self, s):
        if self.redact == "meta":
            return {k: v for k, v in s.items() if k not in ("text", "top")}
        return self.prepare(s)

    @staticmethod
    def headers(h):
        out = {k: v for k, v in h.items() if k.lower() not in DROP_HEADERS}
        if any(k.lower() == "authorization" for k in h.keys()):
            out["authorization"] = "[present, not logged]"
        return out

    def record(self, e):
        """e: one call (see app.py front_door). Thread-safe; files are written by the flush thread."""
        probe = e["path"] == "/" or e["path"].startswith(PROBES)
        e["probe"] = probe
        err = e["status"] >= 400
        with self._lock:
            self.total += 1
            self.errors += err
            r = self.by_route[f"{e['method']} {e['route']}"]
            r["n"] += 1
            r["errors"] += err
            r["ms_sum"] += e["ms"]
            r["ms_max"] = max(r["ms_max"], e["ms"])
            self.by_status[str(e["status"])] += 1
            m = int(e["ts"] // 60 * 60)
            b = self.minutes.setdefault(m, {"t": m, "n": 0, "errors": 0, "ms_sum": 0.0})
            b["n"] += 1
            b["errors"] += err
            b["ms_sum"] += e["ms"]
            while len(self.minutes) > 24 * 60:
                self.minutes.popitem(last=False)
            (self.probes if probe else self.calls).append(e)
            self._dirty = True
            if self.root:
                try:
                    day = time.strftime("%Y-%m-%d", time.gmtime(e["ts"]))
                    with open(os.path.join(self.log_dir, f"{day}.jsonl"), "a") as f:
                        f.write(json.dumps(e, default=str, separators=(",", ":")) + "\n")
                except OSError as x:                     # a full disk must not fail the request
                    self.log("call log: could not append (%s)", x)

    # ---------------------------------------------------------------- reading (tests, /metrics-like use)
    def latest(self, n=50, probes=False):
        with self._lock:
            src = list(self.probes if probes else self.calls)
        return src[-n:][::-1]

    def snapshot(self):
        with self._lock:
            def slim(e):
                return {**e, "request_body": _trim(e.get("request_body"), self.snap_limit),
                        "response_body": _trim(e.get("response_body"), self.snap_limit)}
            return {"server": self.server, "updated_ts": time.time(),
                    "totals": {"calls": self.total, "errors": self.errors, "by_status": dict(self.by_status),
                               "by_route": {k: dict(v) for k, v in self.by_route.items()}},
                    "minutes": list(self.minutes.values())[-180:],
                    "calls": [slim(e) for e in reversed(self.calls)],
                    "probes": [slim(e) for e in reversed(self.probes)]}

    # ---------------------------------------------------------------- files
    def _loop(self):
        while not self._stop.wait(self.flush_every):
            if self._dirty:
                self._flush()

    def _flush(self):
        if not self.root:
            return
        self._dirty = False
        try:
            _atomic_dump(self.snapshot(), os.path.join(self.dir, "servers", f"{self.key}.json"))
        except OSError as x:
            self.log("call log: could not write the snapshot (%s)", x)

    def close(self):
        self._stop.set()
        self._flush()

    def _register(self):
        p = os.path.join(self.dir, "servers.json")
        try:
            items = json.load(open(p))
        except (OSError, ValueError):
            items = []
        items = [s for s in items if isinstance(s, dict) and s.get("key") != self.key]
        items.append({"key": self.key, "path": f"apicalls/servers/{self.key}.json", "started": self.server["started"],
                      "model_id": self.server.get("model_id")})
        _atomic_dump(items, p)

    def _reload(self):
        """Refill the rings from today's (and yesterday's) JSONL, so a restarted server keeps its recent calls."""
        files = sorted(f for f in os.listdir(self.log_dir) if f.endswith(".jsonl"))[-2:]
        rows = []
        for f in files:
            try:
                with open(os.path.join(self.log_dir, f)) as fh:
                    tail = collections.deque(fh, maxlen=self.calls.maxlen + self.probes.maxlen)
                rows += [json.loads(line) for line in tail if line.strip()]
            except (OSError, ValueError):
                continue
        for e in rows[-(self.calls.maxlen + self.probes.maxlen):]:
            (self.probes if e.get("probe") else self.calls).append(e)

    def _prune(self):
        if not self.days:
            return
        cutoff = time.strftime("%Y-%m-%d", time.gmtime(time.time() - self.days * 86400))
        for f in os.listdir(self.log_dir):
            if f.endswith(".jsonl") and f[:10] < cutoff:
                try:
                    os.remove(os.path.join(self.log_dir, f))
                except OSError:
                    pass
