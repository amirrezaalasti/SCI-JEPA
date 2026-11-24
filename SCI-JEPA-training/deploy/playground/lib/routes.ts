// The API routes the playground may call. Keep in step with ALLOWED in src/sci_jepa/api/relay.py
// (the worker checks again, so a mistake here cannot reach anything else).
const SES = "ses_[A-Za-z0-9_-]{1,64}";
const ALLOWED: [string, RegExp][] = [
  ["GET", /^\/v1\/(health|ready|info|models|tasks|papers\/search)$/],
  ["GET", /^\/v1\/papers\/[A-Za-z0-9._-]{1,32}(\/[A-Za-z0-9._-]{1,32})?$/],
  ["GET", /^\/v1\/kus\/ku_\d{1,12}$/],
  ["GET", new RegExp(`^/v1/sessions/${SES}$`)],
  ["DELETE", new RegExp(`^/v1/sessions/${SES}$`)],
  ["POST", /^\/v1\/(ask|ask\/paper|parse|validate|consistency|edit-check|intervention|predict\/(forward|inverse|abduction))$/],
  ["POST", new RegExp(`^/v1/sessions/${SES}/answers$`)],
];
const QUERY_KEYS = new Set(["q", "limit", "cursor"]);
export const MAX_BODY = 64 * 1024;

export function allowed(method: unknown, path: unknown): boolean {
  if (typeof method !== "string" || typeof path !== "string" || path.includes("..") || path.includes("//")) return false;
  return ALLOWED.some(([m, rx]) => m === method && rx.test(path));
}

export function checkQuery(q: unknown): Record<string, string> | null | false {
  if (q == null) return null;
  if (typeof q !== "object" || Array.isArray(q)) return false;
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(q as Record<string, unknown>)) {
    if (!QUERY_KEYS.has(k) || !["string", "number"].includes(typeof v) || String(v).length > 200) return false;
    out[k] = String(v);
  }
  return out;
}
