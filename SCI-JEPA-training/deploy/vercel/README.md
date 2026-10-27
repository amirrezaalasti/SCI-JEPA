# Hosted dashboard (Vercel)

The live training dashboard at one URL for the whole team, behind a shared password. It shows the same
pages as `sci-jepa serve` plus two hosted-only pages: **Notes** (a shared markdown document) and
**Queue** (the project's SLURM jobs).

Production: https://scijepa-dashboard.vercel.app (Vercel project `scijepa-dashboard`)

## How data gets there

```
Grete login node                                    Vercel
runs/paper_a_results/dashboard/  ── push_dashboard.py ──>  /api/push ──> Upstash Redis
  runs.json, runs/*/live.json,      (every 60 s, only                       │
  infra/live.json, graphs/*.html,    changed files, gzip)                   v
  + squeue output                                     browser <── CDN <── /api/manifest, /api/f
```

Compute nodes have no internet, so `scripts/push_dashboard.py` runs on a login node and reads the
dashboard folder that training already writes on the shared file system. Training code and job scripts
are unchanged.

In the browser, `static/hosted.js` answers the unchanged pages' `fetch("runs.json")` and
`fetch("runs/<name>/live.json")` from the synced copies. Every file version has its own URL, so the CDN
serves repeated polls and Redis is read about once per version. This keeps usage inside the free plans.
Long `history` arrays in `live.json` are thinned to 1,500 rows before upload; local files are never changed.

## Run the sync on the cluster

```bash
# once: put these in ~/.bashrc on the login node (values in deploy/vercel/.env.secrets, ask the project lead)
export SCIJEPA_PUSH_URL=https://scijepa-dashboard.vercel.app
export SCIJEPA_PUSH_TOKEN=<PUSH_TOKEN>

# then, from SCI-JEPA-training/, inside tmux so it keeps running
python3 scripts/push_dashboard.py                                  # default folder, every 60 s
python3 scripts/push_dashboard.py /path/to/dashboard --squeue-args "-A <account>"
```

The header shows "synced N s ago". It turns orange after 10 minutes without a push.

## Deploy and settings

```bash
cd SCI-JEPA-training/deploy/vercel
npm install
npm run deploy            # copies the package's dashboard pages into public/, then vercel deploy --prod
```

Redeploy after changing `src/sci_jepa/dashboard/dashboard.html` or `infra.html`, so the hosted copy matches.

Project environment variables (production):

| Variable | Purpose |
|---|---|
| `DASHBOARD_PASSWORD` | the team password; changing it signs everyone out |
| `SESSION_SECRET` | signs the session cookie |
| `PUSH_TOKEN` | the cluster's upload credential (only `/api/push` accepts it) |
| `KV_REST_API_URL`, `KV_REST_API_TOKEN` | set by the Upstash for Redis integration |

Rotate a secret with `vercel env rm NAME production && vercel env add NAME production`, then `npm run deploy`.
