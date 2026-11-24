// GET /api/relay/pull?wait=25&ready=0|1 (relay worker only): note the worker as alive, then hand it the next job,
// waiting up to `wait` s. 200 = one job (JSON), 204 = none. Jobs nobody waits for any more are dropped.
import { isRelay } from "../../lib/auth.js";
import { problem } from "../../lib/http.js";
import { K, blpop, cmd } from "../../lib/redis.js";

const STALE_MS = 150_000;            // the page gives up after about 3 minutes

export async function GET(req: Request) {
  if (!(await isRelay(req))) return problem(401, "Wrong relay token", "unauthorized");
  const u = new URL(req.url);
  const wait = Math.max(1, Math.min(25, Number(u.searchParams.get("wait")) || 25));
  const ready = u.searchParams.get("ready") === "1";
  await cmd("SET", K.worker, JSON.stringify({ ts: Date.now(), ready }), "EX", 90);
  const deadline = Date.now() + wait * 1000;
  for (;;) {
    const left = Math.floor((deadline - Date.now()) / 1000);
    if (left < 1) return new Response(null, { status: 204 });
    const raw = await blpop(K.jobs, left);
    if (!raw) return new Response(null, { status: 204 });
    let job: any;
    try { job = JSON.parse(raw); } catch { continue; }
    if (Date.now() - Number(job.ts || 0) > STALE_MS) continue;
    return new Response(raw, { status: 200, headers: { "Content-Type": "application/json" } });
  }
}
