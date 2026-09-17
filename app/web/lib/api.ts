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

export async function apiGet<T>(path: string): Promise<T> {
  const r = await fetch(path, { cache: "no-store" });
  const body = (await r.json().catch(() => ({}))) as ApiErrorBody;
  if (!r.ok) throw new Error(body.error || `GET ${path} failed (${r.status})`);
  return body as T;
}

export async function apiSend<T>(path: string, method: string, body?: unknown): Promise<T> {
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
  const r = await fetch(path, { method: "POST", body: fd });
  const data = (await r.json().catch(() => ({}))) as ApiErrorBody;
  if (!r.ok) throw new Error(data.error || `POST ${path} failed (${r.status})`);
  return data as T;
}

export async function fetchModels(): Promise<ModelLists> {
  return apiGet<ModelLists>("/api/models");
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
