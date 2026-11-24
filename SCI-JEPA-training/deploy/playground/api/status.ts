// GET /api/status: is a relay worker connected, and is the API behind it ready? (the page's status pill)
import { K, cmd } from "../lib/redis.js";

export async function GET() {
  const raw = await cmd("GET", K.worker);
  if (!raw) return Response.json({ worker: false });
  const w = JSON.parse(raw);
  return Response.json({ worker: true, ready: !!w.ready, age_s: Math.max(0, (Date.now() - w.ts) / 1000) });
}
