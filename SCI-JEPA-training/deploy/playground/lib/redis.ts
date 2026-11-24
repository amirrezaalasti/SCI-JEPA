// Upstash Redis over its REST API, with fetch only. The SDK has no BLPOP, and the relay relies on blocking pops:
// a waiting request costs one command instead of one per poll. Every key starts with "pg:", so the playground can
// share a database with the training dashboard.
type Arg = string | number;

function env(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`${name} is not set in the Vercel project`);
  return v;
}

async function post(path: string, body: unknown, timeoutMs: number): Promise<any> {
  const r = await fetch(env("KV_REST_API_URL") + path, {
    method: "POST",
    headers: { Authorization: `Bearer ${env("KV_REST_API_TOKEN")}` },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  const j = await r.json();
  if (!r.ok || (j && !Array.isArray(j) && j.error)) throw new Error(`redis: ${j?.error || r.status}`);
  return j;
}

export async function cmd(...args: Arg[]): Promise<any> {
  return (await post("", args.map(String), 10_000)).result;
}

/** BLPOP with a timeout in seconds -> the popped value, or null. */
export async function blpop(key: string, seconds: number): Promise<string | null> {
  const j = await post("", ["BLPOP", key, String(seconds)], (seconds + 10) * 1000);
  return j.result ? j.result[1] : null;
}

/** Several commands in one round trip -> their results (each command still counts once). */
export async function pipeline(cmds: Arg[][]): Promise<any[]> {
  const j = await post("/pipeline", cmds.map((c) => c.map(String)), 10_000);
  return (j as { result?: any; error?: string }[]).map((x) => {
    if (x.error) throw new Error(`redis: ${x.error}`);
    return x.result;
  });
}

export const K = {
  jobs: "pg:jobs",
  worker: "pg:worker",
  result: (id: string) => `pg:res:${id}`,
  rate: (client: string, minute: number) => `pg:rl:${client}:${minute}`,
};
