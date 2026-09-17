import { type ChildProcess, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

// app/api is two levels below the repo root, both as source and compiled.
export function getRepoRoot(): string {
  if (process.env.APPLIO_ROOT && fs.existsSync(process.env.APPLIO_ROOT)) {
    return path.resolve(process.env.APPLIO_ROOT);
  }
  return path.resolve(__dirname, "..", "..", "..");
}

export function getPythonBin(): string {
  return process.env.PYTHON_BIN || (process.platform === "win32" ? "python" : "python3");
}

export function getUploadsDir(): string {
  const dir = process.env.UPLOADS_DIR || path.join(getRepoRoot(), "assets", "audios", "_uploads");
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

export function getOutputsDir(): string {
  const dir = process.env.OUTPUTS_DIR || path.join(getRepoRoot(), "assets", "audios");
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

export interface SpawnResult {
  stdout: string;
  stderr: string;
  code: number | null;
}

export function runPythonModule(
  args: string[],
  opts: {
    cwd?: string;
    onData?: (chunk: string, stream: "stdout" | "stderr") => void;
    onSpawn?: (pid?: number) => void;
  } = {},
): Promise<SpawnResult> {
  const cwd = opts.cwd || getRepoRoot();
  return new Promise((resolve, reject) => {
    const child: ChildProcess = spawn(getPythonBin(), args, {
      cwd,
      env: { ...process.env, PYTHONIOENCODING: "utf-8" },
      windowsHide: true,
    });
    opts.onSpawn?.(child.pid);
    let stdout = "";
    let stderr = "";
    child.stdout?.on("data", (d: Buffer) => {
      const s = d.toString();
      stdout += s;
      opts.onData?.(s, "stdout");
    });
    child.stderr?.on("data", (d: Buffer) => {
      const s = d.toString();
      stderr += s;
      opts.onData?.(s, "stderr");
    });
    child.on("error", reject);
    child.on("close", (code) => resolve({ stdout, stderr, code }));
  });
}

export function resolveInsideRepo(p: string): string {
  const root = getRepoRoot();
  const resolved = path.resolve(root, p);
  const rel = path.relative(root, resolved);
  if (rel.startsWith("..") || rel.includes("..")) {
    throw new Error(`Path escapes repo root: ${p}`);
  }
  return resolved;
}

export function resolveUserPath(p: string): string {
  if (!p) return "";
  if (path.isAbsolute(p)) {
    if (!fs.existsSync(p)) throw new Error(`File not found: ${p}`);
    return p;
  }
  return resolveInsideRepo(p);
}
