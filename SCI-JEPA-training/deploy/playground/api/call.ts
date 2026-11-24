// POST /api/call {method, path, query?, body?}: queue one API request for the relay worker on the cluster and wait
// for its answer (up to WAIT_S). 200 = the answer envelope {status, body, request_id, model_id, ms};
// 202 {id} = still running, ask /api/result?id=... ; 4xx/5xx = refused here (problem JSON).
import { clientOf } from "../lib/auth.js";
import { deliver, newId, problem } from "../lib/http.js";
import { K, blpop, cmd, pipeline } from "../lib/redis.js";
import { MAX_BODY, allowed, checkQuery } from "../lib/routes.js";

const RATE = Number(process.env.RATE_PER_MIN || 30);    // requests per minute per visitor address
const MAX_QUEUE = 32;                                   // the API's own GPU queue holds 16
const WAIT_S = 20;

export async function POST(req: Request) {
  const text = await req.text();
  if (text.length > MAX_BODY + 2048) return problem(413, "Request too large", "too_large", `At most ${MAX_BODY / 1024} KB.`);
  let j: any;
  try { j = JSON.parse(text); } catch { return problem(400, "Malformed request", "invalid_json"); }
  const { method, path } = j ?? {};
  if (!allowed(method, path)) return problem(400, "The playground cannot send this request", "not_allowed", `${method} ${path}`);
  const query = checkQuery(j.query);
  if (query === false) return problem(400, "Malformed query", "invalid_query");
  const body = j.body ?? null;
  if (body !== null && (method !== "POST" || typeof body !== "object")) return problem(400, "Malformed body", "invalid_body");

  const client = await clientOf(req);
  const now = Date.now(), rk = K.rate(client, Math.floor(now / 60000));
  const [count, , worker, queued] = await pipeline([["INCR", rk], ["EXPIRE", rk, 70], ["GET", K.worker], ["LLEN", K.jobs]]);
  if (count > RATE)
    return problem(429, "Too many requests", "rate_limited", `At most ${RATE} requests per minute from one address.`,
      { "Retry-After": String(60 - (Math.floor(now / 1000) % 60)) });
  if (!worker)
    return problem(503, "The model server is offline", "offline",
      "The GPU server on the research cluster is not connected right now. Please try again later.", { "Retry-After": "60" });
  if (queued >= MAX_QUEUE)
    return problem(503, "The model is busy", "overloaded", "Many requests are waiting. Please try again shortly.", { "Retry-After": "10" });

  const id = newId();
  await cmd("RPUSH", K.jobs, JSON.stringify({ id, method, path, query, body, client, ts: now }));
  const got = await blpop(K.result(id), WAIT_S);
  return got ? deliver(got) : Response.json({ id, pending: true }, { status: 202 });
}
