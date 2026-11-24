// Build public/ from the playground page in the sci_jepa package (src/sci_jepa/api/playground.html), the same page
// the API serves at /v1/playground. The package page is not modified: the hosted copy only gets a config line that
// switches it to the relay. Run before every deploy (`npm run deploy` does).
//   PLAYGROUND_API_HINT  the API address shown in code snippets (e.g. http://gpu-t2080ti-01:8000)
import { copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(dirname(fileURLToPath(import.meta.url)));
const page = join(here, "..", "..", "src", "sci_jepa", "api", "playground.html");
const out = join(here, "public");
const MARK = "<!-- playground-config -->";

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

let html = readFileSync(page, "utf8");
if (!html.includes(MARK)) throw new Error(`assemble: marker not found in ${page}: ${MARK}`);
const cfg = { mode: "relay", apiHint: process.env.PLAYGROUND_API_HINT || null };
html = html.replace(MARK, `<script>window.PLAYGROUND = ${JSON.stringify(cfg).replace(/</g, "\\u003c")};</script>\n` +
  '<meta name="robots" content="noindex">');
writeFileSync(join(out, "index.html"), html);
copyFileSync(join(here, "static", "login.html"), join(out, "login.html"));
console.log(`assemble: public/index.html (relay, apiHint=${cfg.apiHint}), public/login.html`);
