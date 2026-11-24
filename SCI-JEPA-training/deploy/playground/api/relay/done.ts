// POST /api/relay/done (relay worker only): the API's answer to one job; the waiting /api/call or /api/result
// picks it up. Kept 2 minutes, then dropped.
import { isRelay } from "../../lib/auth.js";
import { JOB_ID, problem } from "../../lib/http.js";
import { K, pipeline } from "../../lib/redis.js";

export async function POST(req: Request) {
  if (!(await isRelay(req))) return problem(401, "Wrong relay token", "unauthorized");
  const text = await req.text();
  if (text.length > 1_200_000) return problem(413, "Answer too large", "too_large");
  let j: any;
  try { j = JSON.parse(text); } catch { return problem(400, "Malformed answer", "invalid_json"); }
  if (!JOB_ID.test(String(j?.id)) || typeof j.status !== "number") return problem(400, "Malformed answer", "invalid_answer");
  await pipeline([["RPUSH", K.result(j.id), text], ["EXPIRE", K.result(j.id), 120]]);
  return new Response(null, { status: 204 });
}
