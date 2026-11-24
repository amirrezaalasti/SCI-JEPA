# Sci-JEPA Playground (Vercel)

A website where researchers try the model without writing code: predict results, find a method, explain results,
check a claim, change one part of a study ("what if"), spot an edited finding, and browse the corpus. Every result has
**Show the API call**, which gives the same request as `curl` or Python, so the page also teaches the API.

Production: https://scijepa-playground.vercel.app (Vercel project `scijepa-playground`, team `devnerds2020s-projects`)

The page is `src/sci_jepa/api/playground.html`. The API serves the same file at `/v1/playground` (and `/` redirects
there), so people on the TIB network can use it on the API server directly, without this website.

## Why a relay

The API runs on a cluster GPU that only the TIB network (or VPN) can reach. Vercel cannot call it, and a public HTTPS
page cannot call a private `http://` address either. So nothing calls the cluster. A worker on the cluster calls out
instead:

```
browser ──POST /api/call──> Vercel ──RPUSH──> Upstash Redis "pg:jobs"
                                                   │ BLPOP (long poll, 25 s)
relay worker on the cluster ──GET /api/relay/pull──┘   (outbound HTTPS only)
   │ forwards the job to http://<node>:8000/v1/...
   └──POST /api/relay/done──> Redis "pg:res:<id>" ──BLPOP──> the waiting /api/call answers the browser
```

- `/api/call` waits up to 20 s for the answer. Longer jobs return `202 {id}`, and the page keeps asking
  `/api/result?id=`. Measured overhead: about 0.8 s per request.
- Blocking pops keep Redis use low: an idle worker costs about two commands every 25 s (around 210k a month). The
  database is the training dashboard's Upstash database, and every key starts with `pg:`.
- Jobs nobody is waiting for any more (older than 150 s) are dropped. Answers expire after 2 minutes.
- When no worker has been seen for 90 s, `/api/call` answers `503 offline` straight away, and the page shows
  "offline".

## Start the relay on the cluster

Run it on a machine that reaches both the API and the internet: the GPU node next to the server, or the login node.

```bash
cd SCI-JEPA-training
export PLAYGROUND_RELAY_TOKEN=<RELAY_TOKEN from deploy/playground/.env.secrets>
.venv/bin/python -m sci_jepa.api.relay --api http://<gpu node>:8000 --relay https://scijepa-playground.vercel.app
```

The worker uses the standard library only. It logs one line per request, and it re-checks the API's `/v1/ready`
every 30 s, which the page's status pill shows. Network errors are retried with back-off. A wrong token stops it with
exit code 2.

## Safety

| Layer | What it does |
|---|---|
| Access code | `ACCESS_CODE` set: researchers enter it once (cookie for 90 days). Remove the variable and redeploy to open the site to anyone with the link. |
| Route list | Only the `/v1` routes the page uses are relayed (`lib/routes.ts`). The worker checks again (`ALLOWED` in `relay.py`) and only ever calls `--api`. `/metrics`, `/v1/docs` and anything else are refused. |
| Rate limits | 30 requests a minute per visitor address on the website (`RATE_PER_MIN`), and 32 queued jobs at most. The worker sends each visitor to the API as its own client (`Bearer playground-<hash>`), so the API's own limits (60 model calls a minute) and GPU queue also apply per visitor. Addresses are stored only as an HMAC. |
| Relay endpoints | `/api/relay/*` need `RELAY_TOKEN`; the cookie is not enough. |
| Logging | The API's call log records relayed calls like any others, with user agent `sci-jepa-playground-relay/1` and the job ID as `X-Request-Id`. The text people type is in it (see `--call-log-redact`). |

## Deploy

```bash
cd SCI-JEPA-training/deploy/playground
npm install
PLAYGROUND_API_HINT=http://<gpu node>:8000 npm run deploy -- --scope devnerds2020s-projects
```

`npm run deploy` copies the package page into `public/index.html` and adds a line that switches it to the relay. The
package file itself is not changed. Redeploy after changing `playground.html`. `PLAYGROUND_API_HINT` is optional: it
is the API address shown in the code snippets (otherwise `http://<server>:8000`).

Environment variables (production):

| Variable | Purpose |
|---|---|
| `KV_REST_API_URL`, `KV_REST_API_TOKEN` | Upstash Redis, the same database as the dashboard |
| `RELAY_TOKEN` | the cluster worker's credential (`PLAYGROUND_RELAY_TOKEN` there) |
| `SESSION_SECRET` | signs the access cookie and the visitor pseudonyms |
| `ACCESS_CODE` | optional; without it the site is public |
| `RATE_PER_MIN` | optional; requests per minute per visitor (default 30) |

The values are in `deploy/playground/.env.secrets` (gitignored). Rotate one with
`vercel env rm NAME production && vercel env add NAME production`, then redeploy.
