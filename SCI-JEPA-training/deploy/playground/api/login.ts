import { COOKIE, checkCode, sessionToken } from "../lib/auth.js";

const DAY = 86400;
const safeNext = (v: string | null) => (v && v.startsWith("/") && !v.startsWith("//") ? v : "/");

export async function POST(req: Request) {
  const form = await req.formData();
  const nextUrl = safeNext(form.get("next") as string | null);
  if (!(await checkCode(String(form.get("code") || "")))) {
    await new Promise((r) => setTimeout(r, 600));          // slow down guessing
    return new Response(null, { status: 303, headers: { Location: `/login.html?e=1&next=${encodeURIComponent(nextUrl)}` } });
  }
  return new Response(null, {
    status: 303,
    headers: {
      Location: nextUrl,
      "Set-Cookie": `${COOKIE}=${await sessionToken()}; Path=/; Max-Age=${90 * DAY}; HttpOnly; Secure; SameSite=Lax`,
    },
  });
}
