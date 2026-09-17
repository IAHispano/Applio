// Typed client for the Express gateway (same-origin /api via Next.js rewrites).

export function errMsg(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export interface ModelLists {
  models: string[];
  indexes: string[];
  audios: string[];
}

export type JobStatus = "queued" | "running" | "done" | "error";

export interface Job {
  id: string;
  type: string;
  status: JobStatus;
  createdAt: string;
  updatedAt: string;
  finishedAt?: string;
  logs: string[];
  result?: Record<string, unknown>;
  error?: string;
  outputFile?: string;
}

interface ApiErrorBody {
  error?: string;
}

interface CacheEntry {
  data: unknown;
  timestamp: number;
}
const apiCache = new Map<string, CacheEntry>();

export function clearApiCache(pathPrefix?: string) {
  if (!pathPrefix) {
    apiCache.clear();
    return;
  }
  for (const key of apiCache.keys()) {
    if (key.startsWith(pathPrefix)) apiCache.delete(key);
  }
}

function invalidateFor(path: string) {
  if (path.includes("/models")) clearApiCache("/api/models");
  if (path.includes("/presets")) clearApiCache("/api/presets");
  if (path.includes("/train")) clearApiCache("/api/train");
  if (path.includes("/settings")) clearApiCache("/api/settings");
  if (path.includes("/plugins")) clearApiCache("/api/plugins");
  if (path.includes("/download")) {
    clearApiCache("/api/models");
    clearApiCache("/api/train");
  }
}

export async function apiGet<T>(path: string, options?: { ttlMs?: number; force?: boolean }): Promise<T> {
  const isJob = path.includes("/jobs");
  const ttl = options?.ttlMs ?? (isJob ? 0 : 30000);
  const now = Date.now();

  if (!options?.force && ttl > 0 && apiCache.has(path)) {
    const entry = apiCache.get(path);
    if (entry && now - entry.timestamp < ttl) {
      return entry.data as T;
    }
  }

  const r = await fetch(path, { cache: "no-store" });
  const body = (await r.json().catch(() => ({}))) as ApiErrorBody;
  if (!r.ok) throw new Error(body.error || `GET ${path} failed (${r.status})`);

  if (ttl > 0) {
    apiCache.set(path, { data: body, timestamp: now });
  }
  return body as T;
}

export async function apiSend<T>(path: string, method: string, body?: unknown): Promise<T> {
  invalidateFor(path);
  const r = await fetch(path, {
    method,
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = (await r.json().catch(() => ({}))) as ApiErrorBody;
  if (!r.ok) throw new Error(data.error || `${method} ${path} failed (${r.status})`);
  return data as T;
}

export async function postForm<T>(path: string, fd: FormData): Promise<T> {
  invalidateFor(path);
  const r = await fetch(path, { method: "POST", body: fd });
  const data = (await r.json().catch(() => ({}))) as ApiErrorBody;
  if (!r.ok) throw new Error(data.error || `POST ${path} failed (${r.status})`);
  return data as T;
}

export async function fetchModels(force = false): Promise<ModelLists> {
  return apiGet<ModelLists>("/api/models", { force });
}

export async function submitJob(path: string, body: unknown): Promise<{ jobId: string }> {
  if (typeof FormData !== "undefined" && body instanceof FormData) return postForm(path, body);
  return apiSend(path, "POST", body);
}

export async function submitInference(fd: FormData): Promise<{ jobId: string }> {
  return postForm("/api/inference", fd);
}

export async function fetchJob(id: string): Promise<{ job: Job }> {
  return apiGet(`/api/jobs/${id}`);
}

export async function stopJob(id: string): Promise<void> {
  await apiSend(`/api/jobs/${id}/stop`, "POST");
}

export function pollJob(id: string, onUpdate: (job: Job) => void): () => void {
  let stopped = false;
  let timer: ReturnType<typeof setInterval>;
  const tick = async () => {
    try {
      const { job } = await fetchJob(id);
      onUpdate(job);
      if (job.status === "done" || job.status === "error") {
        clearInterval(timer);
      }
    } catch {
      /* keep polling through transient proxy restarts */
    }
  };
  timer = setInterval(() => {
    if (!stopped) void tick();
  }, 2000);
  void tick();
  const timeout = setTimeout(
    () => {
      stopped = true;
      clearInterval(timer);
    },
    12 * 60 * 60 * 1000,
  );
  return () => {
    stopped = true;
    clearInterval(timer);
    clearTimeout(timeout);
  };
}

export function outputUrl(rel: string): string {
  return `/outputs/${rel.split("/").pop()}`;
}

export function isAudioFile(rel: string): boolean {
  return [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"].some((e) => rel.toLowerCase().endsWith(e));
}

export function isImageFile(rel: string): boolean {
  return [".png", ".jpg", ".jpeg"].some((e) => rel.toLowerCase().endsWith(e));
}
