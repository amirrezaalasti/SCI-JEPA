// path -> version for every synced file. Viewers poll this; the CDN answers most polls.
import { MANIFEST, redis } from "../lib/store.js";

export async function GET() {
  return Response.json((await redis.hgetall(MANIFEST)) || {}, {
    headers: { "Cache-Control": "public, max-age=0, s-maxage=10, stale-while-revalidate=20" },
  });
}
