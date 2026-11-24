// With ACCESS_CODE set, every page and /api/* except the sign-in and the relay worker's endpoints needs the cookie.
// Without it, the site is open to anyone who has the link.
import { next } from "@vercel/functions";
import { accessRequired, isAuthed } from "./lib/auth.js";

export const config = {
  matcher: ["/((?!login\\.html|api/login|api/relay/|favicon\\.ico).*)"],
};

export default async function middleware(req: Request) {
  if (!accessRequired() || (await isAuthed(req))) return next();
  const url = new URL(req.url);
  if (url.pathname.startsWith("/api/"))
    return Response.json({ title: "Sign in first", status: 401, code: "unauthorized" }, { status: 401 });
  const back = encodeURIComponent(url.pathname + url.search + url.hash);
  return Response.redirect(new URL(`/login.html?next=${back}`, req.url), 302);
}
