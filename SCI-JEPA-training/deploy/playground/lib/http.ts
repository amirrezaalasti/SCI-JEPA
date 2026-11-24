// RFC 9457 problem answers, in the same shape as the API's own errors, so the page shows them the same way.
export function problem(status: number, title: string, code: string, detail?: string, headers: Record<string, string> = {}) {
  return Response.json(
    { type: `https://sci-jepa.org/problems/${code}`, title, status, code, detail: detail ?? null },
    { status, headers: { "Content-Type": "application/problem+json", ...headers } },
  );
}

export const JOB_ID = /^pg_[0-9a-f]{16}$/;

export function newId(): string {
  const b = crypto.getRandomValues(new Uint8Array(8));
  return "pg_" + Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
}

/** The relay's answer envelope as stored in Redis -> the response the page gets. */
export function deliver(stored: string): Response {
  return new Response(stored, { status: 200, headers: { "Content-Type": "application/json" } });
}
