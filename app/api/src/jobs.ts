import { v4 as uuidv4 } from "uuid";

export type JobStatus = "queued" | "running" | "done" | "error";
export type JobType = "inference" | "batch-inference" | "train" | "tts" | "download" | "other";

export interface Job {
  id: string;
  type: JobType;
  status: JobStatus;
  createdAt: string;
  updatedAt: string;
  finishedAt?: string;
  params?: Record<string, unknown>;
  logs: string[];
  result?: Record<string, unknown>;
  error?: string;
  outputFile?: string; // repo-relative path served under /outputs
}

const jobs = new Map<string, Job>();
const MAX_JOBS = 200;

export function createJob(type: JobType, params?: Record<string, unknown>): Job {
  const now = new Date().toISOString();
  const job: Job = {
    id: uuidv4(),
    type,
    status: "queued",
    createdAt: now,
    updatedAt: now,
    params,
    logs: [],
  };
  jobs.set(job.id, job);
  if (jobs.size > MAX_JOBS) {
    const oldest = [...jobs.values()].sort((a, b) => a.createdAt.localeCompare(b.createdAt))[0];
    if (oldest) jobs.delete(oldest.id);
  }
  return job;
}

export function getJob(id: string): Job | undefined {
  return jobs.get(id);
}

export function listJobs(): Job[] {
  return [...jobs.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export function appendLog(job: Job, line: string) {
  job.logs.push(line.slice(0, 2000));
  if (job.logs.length > 500) job.logs = job.logs.slice(-500);
  job.updatedAt = new Date().toISOString();
}

export function setRunning(job: Job) {
  job.status = "running";
  job.updatedAt = new Date().toISOString();
}

export function setDone(job: Job, result?: Record<string, unknown>, outputFile?: string) {
  job.status = "done";
  job.result = result;
  if (outputFile) job.outputFile = outputFile;
  job.finishedAt = new Date().toISOString();
  job.updatedAt = job.finishedAt;
}

export function setError(job: Job, error: string) {
  job.status = "error";
  job.error = error;
  job.finishedAt = new Date().toISOString();
  job.updatedAt = job.finishedAt;
}
