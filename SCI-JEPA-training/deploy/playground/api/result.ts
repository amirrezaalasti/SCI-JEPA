// GET /api/result?id=pg_...: keep waiting for an answer that /api/call did not get in time.
import { JOB_ID, deliver, problem } from "../lib/http.js";
import { K, blpop } from "../lib/redis.js";

export async function GET(req: Request) {
  const id = new URL(req.url).searchParams.get("id") || "";
  if (!JOB_ID.test(id)) return problem(400, "Unknown request id", "invalid_id");
  const got = await blpop(K.result(id), 20);
  return got ? deliver(got) : Response.json({ id, pending: true }, { status: 202 });
}
