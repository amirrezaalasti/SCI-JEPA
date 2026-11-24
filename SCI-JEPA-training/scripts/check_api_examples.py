"""
Send every example of the API documentation (/v1/openapi.json) to a running server and compare the answers
with the documented ones: the status code, and the shape of the body (the same keys and types at every level;
the values, such as scores or session IDs, may differ).

    python scripts/check_api_examples.py --url http://<node>:8000             # check
    python scripts/check_api_examples.py --url http://<node>:8000 --write     # record the answers as the docs

The examples are read from the documentation the local code builds (sci_jepa.api.examples), or with
--docs-url from a server's /v1/openapi.json. --write stores the answers in
src/sci_jepa/api/examples_responses.json; restart the server so /v1/docs shows them.
Exit code 0 when every example passes.
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from types import SimpleNamespace


def local_openapi():
    from sci_jepa.api.app import create_app
    eng = SimpleNamespace(lock=threading.Lock(), index_spec="none",
                          state={"model_loaded": False, "error": None, "index_loaded": False, "search_loaded": False})
    return create_app(eng, load_in_background=False).openapi()


def call(base, method, path, query=None, body=None, timeout=300):
    url = base.rstrip("/") + path + ("?" + urllib.parse.urlencode(query) if query else "")
    data = json.dumps(body).encode() if body is not None else None
    for _ in range(5):                                   # wait out the server's rate limit (429 + Retry-After)
        req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw, status = r.read(), r.status
        except urllib.error.HTTPError as e:
            raw, status = e.read(), e.code
            if status == 429:
                wait = int(e.headers.get("Retry-After", "10")) + 1
                print(f"     rate limited, waiting {wait} s")
                time.sleep(wait)
                continue
        break
    return status, (json.loads(raw) if raw else None)


def shape_diff(doc, got, where="body"):
    """Differences in keys and types between the documented and the received answer."""
    if doc is None or got is None:
        return []
    if isinstance(doc, dict):
        if not isinstance(got, dict):
            return [f"{where}: documented an object, got {type(got).__name__}"]
        out = [f"{where}.{k}: documented, missing in the answer" for k in doc if k not in got]
        out += [f"{where}.{k}: in the answer, not documented" for k in got if k not in doc]
        for k in doc.keys() & got.keys():
            out += shape_diff(doc[k], got[k], f"{where}.{k}")
        return out
    if isinstance(doc, list):
        if not isinstance(got, list):
            return [f"{where}: documented a list, got {type(got).__name__}"]
        return shape_diff(doc[0], got[0], f"{where}[0]") if doc and got else []
    num = (int, float)
    if isinstance(doc, bool) or isinstance(got, bool):
        return [] if isinstance(doc, bool) == isinstance(got, bool) else [f"{where}: {doc!r} vs {got!r}"]
    if isinstance(doc, num) and isinstance(got, num) or type(doc) is type(got):
        return []
    return [f"{where}: documented {type(doc).__name__}, got {type(got).__name__}"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="the server, e.g. http://gpu-t2080ti-01:8000")
    ap.add_argument("--docs-url", help="read the examples from this server's /v1/openapi.json instead of the local code")
    ap.add_argument("--write", action="store_true", help="record the answers as the documented responses")
    a = ap.parse_args()

    from sci_jepa.api import examples as X
    spec = call(a.docs_url, "GET", "/v1/openapi.json")[1] if a.docs_url else local_openapi()
    name_of = {(m, p): n for n, (m, p) in X.ROUTES.items()}
    recorded, failures, n = {}, [], 0

    def new_session():
        _, b = call(a.url, "POST", "/v1/ask", body=X.REQUESTS["ask"]["need_more"][1])
        return b["session"]

    def check(label, name, ex_name, status, want_status, got, documented):
        nonlocal n
        n += 1
        problems = [] if status == want_status else [f"status {status}, documented {want_status}: "
                                                     f"{json.dumps(got)[:300]}"]
        if documented is not None and status == want_status:
            problems += shape_diff(documented, got)
        print(f"{'ok  ' if not problems else 'FAIL'} {label}  [{ex_name}]  {status}")
        for p in problems:
            print("       " + p)
        if problems:
            failures.append(label)
        if status == want_status and got is not None and name != "_errors":
            recorded.setdefault(name, {})[ex_name] = got

    order = sorted(spec["paths"].items(), key=lambda kv: "session" in kv[0])   # sessions last: they need /v1/ask
    for path, ops in order:
        for method, op in ops.items():
            method = method.upper()
            name = name_of.get((method, path))
            if name is None:
                continue
            docs_200 = (op.get("responses", {}).get("200", {}).get("content", {})
                        .get("application/json", {}).get("examples", {}))
            body_ex = (op.get("requestBody", {}).get("content", {}).get("application/json", {})
                       .get("openapi_examples") or op.get("requestBody", {}).get("content", {})
                       .get("application/json", {}).get("examples"))
            params = {p["name"]: (p["in"], (p.get("examples") or {}).get("example", {}).get("value"))
                      for p in op.get("parameters", [])}
            if method == "POST" and not body_ex:
                print(f"FAIL {method} {path}: no request example in the docs")
                failures.append(f"{method} {path}")
                continue
            for ex_name, ex in (body_ex or {"default": {"value": None}}).items():
                p, query = path, {}
                for k, (where, v) in params.items():
                    if k == "session_id":
                        v = new_session()
                    if v is None:
                        continue
                    if where == "path":
                        p = p.replace("{" + k + "}", str(v))
                    else:
                        query[k] = v
                status, got = call(a.url, method, p, query, ex["value"])
                documented = (docs_200.get(ex_name) or {}).get("value")
                check(f"{method} {path}", name, ex_name, status, 204 if method == "DELETE" else 200, got, documented)

    # the documented error answers, per endpoint
    for name, by_code in X.ERRORS.items():
        method, path = X.ROUTES[name]
        op = spec["paths"].get(path, {}).get(method.lower(), {})
        for code, (summary, req) in by_code.items():
            p = path
            for k, v in (req.get("path") or {}).items():
                p = p.replace("{" + k + "}", new_session() if v is None else str(v))
            status, got = call(a.url, method, p, req.get("query"), req.get("body"))
            doc = (op.get("responses", {}).get(code, {}).get("content", {}).get("application/problem+json", {})
                   .get("examples", {}).get(code, {}).get("value"))
            check(f"{method} {path}", "_errors", f"error {code}", status, int(code), got, doc)
            if status == int(code):
                recorded.setdefault("_errors", {}).setdefault(name, {})[code] = got

    print(f"\n{n - len(failures)}/{n} examples pass")
    if a.write:
        X.RESPONSES_FILE.write_text(json.dumps(recorded, indent=1, ensure_ascii=False) + "\n")
        print(f"wrote {X.RESPONSES_FILE}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
