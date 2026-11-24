// Two credentials:
// - ACCESS_CODE (optional): the code researchers type once. Without it the site is open to anyone with the link.
//   The cookie is HMAC(SESSION_SECRET, ACCESS_CODE), so changing either signs everyone out.
// - RELAY_TOKEN: the relay worker on the cluster (`Authorization: Bearer ...`), for /api/relay/* only.
// Web Crypto only, so the same code runs in the routing middleware and in the functions.
export const COOKIE = "pg_session";
const enc = new TextEncoder();

function env(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`${name} is not set in the Vercel project`);
  return v;
}

export async function hmacHex(key: string, msg: string): Promise<string> {
  const k = await crypto.subtle.importKey("raw", enc.encode(key), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = new Uint8Array(await crypto.subtle.sign("HMAC", k, enc.encode(msg)));
  return Array.from(sig, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Constant-time comparison (hash both sides first so the lengths match). */
export async function same(a: string, b: string): Promise<boolean> {
  const [x, y] = await Promise.all([hmacHex("cmp", a), hmacHex("cmp", b)]);
  let d = 0;
  for (let i = 0; i < x.length; i++) d |= x.charCodeAt(i) ^ y.charCodeAt(i);
  return d === 0;
}

export const accessRequired = () => !!process.env.ACCESS_CODE;
export const sessionToken = () => hmacHex(env("SESSION_SECRET"), "pg:" + env("ACCESS_CODE"));
export const checkCode = (given: string) => same(given.trim(), env("ACCESS_CODE"));

function cookie(req: Request, name: string): string | null {
  for (const part of (req.headers.get("cookie") || "").split(";")) {
    const i = part.indexOf("=");
    if (i > 0 && part.slice(0, i).trim() === name) return decodeURIComponent(part.slice(i + 1).trim());
  }
  return null;
}

export async function isAuthed(req: Request): Promise<boolean> {
  if (!accessRequired()) return true;
  const c = cookie(req, COOKIE);
  return !!c && same(c, await sessionToken());
}

export async function isRelay(req: Request): Promise<boolean> {
  const h = req.headers.get("authorization") || "";
  return h.startsWith("Bearer ") && same(h.slice(7), env("RELAY_TOKEN"));
}

/** A stable pseudonym for the visitor's address: rate limits here and in the API, never the address itself. */
export async function clientOf(req: Request): Promise<string> {
  const ip = (req.headers.get("x-forwarded-for") || "").split(",")[0].trim() || req.headers.get("x-real-ip") || "unknown";
  return (await hmacHex(env("SESSION_SECRET"), "client:" + ip)).slice(0, 16);
}
