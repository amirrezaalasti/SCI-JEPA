"""
Serve a trained SCI-JEPA model over HTTP (FastAPI, first version, no authentication).

    sci-jepa api --model <model folder> [--index auto|none|<paper_a_ku_index_*.pt>] [--port 8000]
    # same as: python -m sci_jepa.api ...

<model folder>: written by training (train.save_model): config.json + model.safetensors
(+ best.safetensors). --index auto finds the KU index the model was trained on under
$PAPER_A_INDEX_DIR (else the index_dir in the model's config). The first start with a full corpus
computes one vector per paper in a single pass over the KU vectors and caches it next to the index
(or in --cache-dir); later starts read the cache. /v1/ready answers 503 until everything is loaded.

Interactive docs: http://<host>:<port>/v1/docs

Call log: with --dashboard <folder> (e.g. <results>/dashboard), every call with its request and answer is
written there for the dashboard's "API calls" tab (apicalls.html; sci_jepa/api/calllog.py), and pushed to the
hosted dashboard when SCIJEPA_PUSH_URL and SCIJEPA_PUSH_TOKEN are set. --dashboard-port also serves the folder.
"""
import argparse
import logging
import os


def main(argv=None):
    ap = argparse.ArgumentParser("sci-jepa api", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="trained model folder (config.json + model.safetensors)")
    ap.add_argument("--weights", default="model", choices=["model", "best"], help="end of training or best validation epoch")
    ap.add_argument("--index", default="auto", help="'auto', 'none' (model-only: consistency, edit-check) or an index .pt")
    ap.add_argument("--cache-dir", default=os.environ.get("SCIJEPA_API_CACHE"), help="where to cache the paper vectors")
    ap.add_argument("--full-texts", action="store_true",
                    help="load the full KU texts (side file of the index; large) instead of 240-character snippets")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1")
    ap.add_argument("--shortlist", type=int, default=64, help="papers whose KUs are scored per request")
    ap.add_argument("--no-neighbours", action="store_true", help="encode the user's paper alone (no corpus context)")
    ap.add_argument("--version", default="1.0.0", help="model version shown as sci-jepa-<version>")
    ap.add_argument("--session-ttl", type=int, default=24 * 3600, help="seconds before a session and its text are deleted")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--licence", default=os.environ.get("SCIJEPA_API_LICENCE"), help="licence shown in /v1/info")
    ap.add_argument("--citation", default=os.environ.get("SCIJEPA_API_CITATION"), help="citation shown in /v1/info")
    ap.add_argument("--released", help="release date shown in /v1/models (default: date of the weights file)")
    g = ap.add_argument_group("robustness (sci_jepa/api/guard.py)")
    g.add_argument("--gpu-queue", type=int, default=16, help="requests waiting for the GPU before 503 overloaded")
    g.add_argument("--gpu-wait", type=float, default=30.0, help="seconds a request may wait for the GPU before 503")
    g.add_argument("--rate-gpu", type=int, default=60, help="model requests per minute and client (0 = no limit)")
    g.add_argument("--rate-read", type=int, default=600, help="other requests per minute and client (0 = no limit)")
    g.add_argument("--max-body", type=int, default=64 * 1024, help="largest request body in bytes (413 above)")
    g.add_argument("--max-sessions", type=int, default=10_000, help="open /v1/ask sessions held in memory")
    g.add_argument("--trust-proxy", action="store_true",
                   help="take the client address from X-Forwarded-For (only behind your own reverse proxy)")
    g.add_argument("--cors-origin", action="append", default=[], help="allowed browser origin (repeatable)")
    g.add_argument("--exit-on-cuda-fault", action="store_true",
                   help="exit with code 3 after a CUDA fault so a supervisor (the sbatch loop) restarts the server")
    c = ap.add_argument_group("call log (sci_jepa/api/calllog.py, dashboard: API calls)")
    c.add_argument("--dashboard", default=os.environ.get("SCIJEPA_API_DASHBOARD"),
                   help="dashboard folder the call log is written to (env SCIJEPA_API_DASHBOARD; default: memory only)")
    c.add_argument("--dashboard-port", type=int, default=None,
                   help="also serve the dashboard folder on 127.0.0.1:<port> (default: not served)")
    c.add_argument("--call-log-redact", default="full", choices=["full", "pii", "meta"],
                   help="full: bodies as sent; pii: emails and phone numbers removed; meta: no bodies")
    c.add_argument("--call-log-days", type=int, default=30, help="days of JSONL call history kept (0 = forever)")
    c.add_argument("--call-log-keep", type=int, default=300, help="latest calls shown on the dashboard")
    a = ap.parse_args(argv)

    import uvicorn
    from sci_jepa.api import textparts
    from sci_jepa.api.app import create_app
    from sci_jepa.api.calllog import CallLog
    from sci_jepa.api.engine import Engine
    from sci_jepa.api.guard import Limits
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("sci_jepa.api")
    eng = Engine(a.model, weights=a.weights, device=a.device, index=a.index, cache_dir=a.cache_dir,
                 full_texts=a.full_texts, shortlist_papers=a.shortlist, neighbours=not a.no_neighbours,
                 version=a.version, log=log.info, licence=a.licence, citation=a.citation, released=a.released)
    limits = Limits(max_body_bytes=a.max_body, gpu_queue=a.gpu_queue, gpu_wait_s=a.gpu_wait, rate_gpu=a.rate_gpu,
                    rate_read=a.rate_read, trust_proxy=a.trust_proxy, max_sessions=a.max_sessions,
                    exit_on_cuda_fault=a.exit_on_cuda_fault, cors_origins=tuple(a.cors_origin),
                    idempotency_ttl_s=a.session_ttl)
    calls = None
    if a.dashboard:
        from sci_jepa.dashboard.push import start_autopush
        from sci_jepa.dashboard.serve import setup_dashboard, start_server
        setup_dashboard(a.dashboard)
        calls = CallLog(a.dashboard, host=a.host if a.host not in ("0.0.0.0", "::") else None, port=a.port,
                        keep=a.call_log_keep, redact=a.call_log_redact, days=a.call_log_days,
                        scrub=textparts.scrub_pii, log=log.warning,
                        server={"model": a.model, "weights": a.weights, "index": a.index, "version": a.version})
        log.info("call log: %s/apicalls (redact=%s)", a.dashboard, a.call_log_redact)
        if a.dashboard_port:
            log.info("dashboard: %sapicalls.html", start_server(a.dashboard, a.dashboard_port))
        start_autopush(a.dashboard)                      # hosted dashboard, if SCIJEPA_PUSH_URL/TOKEN are set
    # One worker process: the model and its GPU memory exist once. limit_concurrency makes uvicorn itself answer
    # 503 when far more connections are open than the GPU queue can take; idle keep-alive connections close fast.
    uvicorn.run(create_app(eng, session_ttl=a.session_ttl, limits=limits, call_log=calls), host=a.host, port=a.port,
                log_level="info", workers=1, limit_concurrency=max(64, 4 * a.gpu_queue), timeout_keep_alive=5,
                timeout_graceful_shutdown=30, proxy_headers=a.trust_proxy,
                forwarded_allow_ips="*" if a.trust_proxy else None)


if __name__ == "__main__":
    main()
