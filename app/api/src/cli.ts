import { spawn } from "node:child_process";
import path from "node:path";
import { errMsg } from "./errors";
import { appendLog, createJob, getJob, type Job, type JobType, setDone, setError, setRunning } from "./jobs";
import { getRepoRoot, runPythonModule } from "./python";

const jobPids = new Map<string, number>();

export function trackPid(jobId: string, pid?: number) {
  if (pid) jobPids.set(jobId, pid);
  else jobPids.delete(jobId);
}

// Best-effort kill of a tracked job's process tree.
export function killJobTree(jobId: string): boolean {
  const pid = jobPids.get(jobId);
  if (!pid) return false;
  try {
    if (process.platform === "win32") {
      spawn("taskkill", ["/PID", String(pid), "/T", "/F"], { windowsHide: true });
    } else {
      process.kill(pid, "SIGTERM");
      setTimeout(() => {
        try {
          process.kill(pid, 0);
          process.kill(pid, "SIGKILL");
        } catch {
          /* already dead */
        }
      }, 5000).unref?.();
    }
    return true;
  } catch {
    return false;
  }
}

export interface CliJobOptions {
  parse?: (stdout: string, stderr: string) => { result?: Record<string, unknown>; outputFile?: string };
}

// Spawns `python <args>` as a tracked job and returns immediately (202 {jobId}).
export function startCliJob(
  type: JobType,
  params: Record<string, unknown>,
  args: string[],
  opts: CliJobOptions = {},
): Job {
  const job = createJob(type, params);
  void (async () => {
    setRunning(job);
    try {
      const r = await runPythonModule(args, {
        onData: (chunk) => {
          const trimmed = chunk.trim().slice(0, 1000);
          if (trimmed) appendLog(job, trimmed);
        },
        onSpawn: (pid) => trackPid(job.id, pid),
      });
      trackPid(job.id, undefined);
      if (r.code !== 0) {
        throw new Error(r.stderr.slice(-3000) || `Process exited with code ${r.code}`);
      }
      const parsed = opts.parse?.(r.stdout, r.stderr);
      setDone(job, parsed?.result ?? { message: r.stdout.trim().split("\n").pop() }, parsed?.outputFile);
    } catch (err) {
      trackPid(job.id, undefined);
      appendLog(job, `ERROR: ${errMsg(err)}`);
      const j = getJob(job.id);
      if (j && j.status === "running") setError(j, errMsg(err) || "Job failed");
    }
  })();
  return job;
}

// Runs `python -c <code>` where code prints one `APPLIO_JSON:{...}` line.
export async function runPythonJson<T = unknown>(code: string, onData?: (line: string) => void): Promise<T> {
  const r = await runPythonModule(["-c", code], {
    onData: (chunk, stream) => {
      if (stream === "stderr") onData?.(`[stderr] ${chunk.trim().slice(0, 500)}`);
    },
  });
  if (r.code !== 0) throw new Error(r.stderr.slice(-3000) || "Python failed");
  const marker = r.stdout.split("\n").find((l) => l.startsWith("APPLIO_JSON:"));
  if (!marker) throw new Error(`Python did not return JSON: ${r.stdout.slice(-500)}`);
  return JSON.parse(marker.slice("APPLIO_JSON:".length)) as T;
}

export function repoRel(absPath: string): string {
  return path.relative(getRepoRoot(), absPath).replace(/\\/g, "/");
}
