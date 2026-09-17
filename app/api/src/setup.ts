import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { appendLog, createJob, getJob, type Job, setDone, setError, setRunning } from "./jobs";
import { getRepoRoot } from "./python";

// First-run setup engine: checks every dependency on startup and installs
// what's missing, streaming progress as a job.

export type CheckState = "ok" | "missing" | "warn";

export interface SetupCheck {
  id: string;
  label: string;
  status: CheckState;
  detail: string;
}

export interface SetupStatus {
  ready: boolean;
  checks: SetupCheck[];
  python: string[] | null;
  checkedAt: string;
}

export interface PythonInfo {
  cmd: string[];
  version: string;
  source: string;
}

const MIN_NODE = 20;
const SLOW_CHECK_TTL_MS = 60 * 60 * 1000;

let cached: { at: number; status: SetupStatus } | null = null;
let activeInstallId: string | null = null;

function exists(p: string): boolean {
  try {
    return fs.existsSync(p);
  } catch {
    return false;
  }
}

interface RunResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

function runCmd(
  cmd: string,
  args: string[],
  opts: { timeoutMs?: number; cwd?: string; shell?: boolean } = {},
): Promise<RunResult> {
  return new Promise((resolve) => {
    let done = false;
    const cwd = opts.cwd || getRepoRoot();
    const pathEnv = `${cwd}${path.delimiter}${process.env.PATH || ""}`;
    const child = spawn(cmd, args, {
      cwd,
      windowsHide: true,
      shell: opts.shell || false,
      env: { ...process.env, PATH: pathEnv, PYTHONIOENCODING: "utf-8" },
    });
    let stdout = "";
    let stderr = "";
    const finish = (code: number | null) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve({ code, stdout, stderr });
    };
    const timer = setTimeout(() => {
      try {
        child.kill();
      } catch {
        /* already dead */
      }
      finish(124);
    }, opts.timeoutMs || 120000);
    timer.unref?.();
    child.stdout?.on("data", (d: Buffer) => {
      stdout += d.toString();
    });
    child.stderr?.on("data", (d: Buffer) => {
      stderr += d.toString();
    });
    child.on("error", () => finish(127));
    child.on("close", (code) => finish(code));
  });
}

function parsePyVersion(out: string): string | null {
  const m = out.match(/Python\s+(\d+)\.(\d+)\.(\d+)/);
  return m ? `${m[1]}.${m[2]}.${m[3]}` : null;
}

function pySupported(version: string): boolean {
  const m = version.match(/(\d+)\.(\d+)/);
  if (!m) return false;
  return Number(m[1]) === 3 && Number(m[2]) >= 10 && Number(m[2]) <= 12;
}

export async function findPython(): Promise<PythonInfo | null> {
  const root = getRepoRoot();
  const candidates: Array<{ cmd: string[]; source: string }> = [];
  if (process.env.PYTHON_BIN) candidates.push({ cmd: [process.env.PYTHON_BIN], source: "PYTHON_BIN" });
  if (process.platform === "win32") {
    candidates.push({ cmd: [path.join(root, "env", "python.exe")], source: "app env/" });
    candidates.push({ cmd: [path.join(root, ".venv", "Scripts", "python.exe")], source: "app .venv" });
    candidates.push({ cmd: ["py", "-3.12"], source: "py launcher" });
    candidates.push({ cmd: ["py", "-3.11"], source: "py launcher" });
    candidates.push({ cmd: ["py", "-3"], source: "py launcher" });
    candidates.push({ cmd: ["python"], source: "PATH" });
  } else {
    candidates.push({ cmd: [path.join(root, "env", "bin", "python")], source: "app env/" });
    candidates.push({ cmd: [path.join(root, ".venv", "bin", "python")], source: "app .venv" });
    candidates.push({ cmd: ["python3"], source: "PATH" });
    candidates.push({ cmd: ["python"], source: "PATH" });
  }
  for (const c of candidates) {
    if (path.isAbsolute(c.cmd[0]) && !exists(c.cmd[0])) continue;
    const r = await runCmd(c.cmd[0], [...c.cmd.slice(1), "--version"], { timeoutMs: 15000 });
    const version = parsePyVersion(r.stdout + r.stderr);
    if (r.code === 0 && version && pySupported(version)) {
      return { cmd: c.cmd, version, source: c.source };
    }
  }
  return null;
}

async function checkEngineDeps(py: string[]): Promise<{ ok: boolean; detail: string }> {
  const code = "import torch, uvicorn, librosa; print(torch.__version__)";
  const r = await runCmd(py[0], [...py.slice(1), "-c", code], { timeoutMs: 180000 });
  if (r.code === 0) return { ok: true, detail: `torch ${r.stdout.trim()}` };
  return {
    ok: false,
    detail: (r.stderr.trim().split("\n").pop() || "engine packages missing").slice(0, 300),
  };
}

async function checkFfmpeg(): Promise<{ ok: boolean; detail: string }> {
  const root = getRepoRoot();
  const exeName = process.platform === "win32" ? "ffmpeg.exe" : "ffmpeg";
  const localExe = path.join(root, exeName);
  const exe = exists(localExe) ? localExe : exeName;
  const r = await runCmd(exe, ["-version"], { timeoutMs: 15000 });
  if (r.code === 0) {
    const detail = (r.stdout + r.stderr).split("\n")[0].trim().slice(0, 120);
    return { ok: true, detail: exists(localExe) ? `${detail} (bundled)` : detail };
  }
  return { ok: false, detail: "ffmpeg not on PATH" };
}

const WEB_PORT = process.env.WEB_PORT || "3000";

// A running `next dev` owns app/web/.next (it keeps .next/trace open), so a
// concurrent `next build` dies with EPERM on Windows. Probe the port instead
// of failing the whole install over a build dev mode does not need.
async function webDevServerRunning(): Promise<boolean> {
  try {
    const res = await fetch(`http://127.0.0.1:${WEB_PORT}/`, {
      method: "HEAD",
      signal: AbortSignal.timeout(1500),
    });
    return res.status < 500;
  } catch {
    return false;
  }
}

function checkWebBuild(): { ok: boolean; detail: string } {
  const root = getRepoRoot();
  if (exists(path.join(root, "app", "web", ".next", "standalone", "server.js"))) {
    return { ok: true, detail: "production build ready" };
  }
  if (exists(path.join(root, "app", "web", ".next", "BUILD_ID"))) {
    return { ok: true, detail: "build ready" };
  }
  if (exists(path.join(root, "app", "web", "package.json"))) {
    return { ok: true, detail: "web source ready" };
  }
  return { ok: false, detail: "web bundle missing — run npm run build" };
}

export async function getStatus(force = false): Promise<SetupStatus> {
  if (!force && cached && Date.now() - cached.at < SLOW_CHECK_TTL_MS) return cached.status;

  const checks: SetupCheck[] = [];
  const nodeMajor = Number(process.version.replace(/^v/, "").split(".")[0]);
  checks.push({
    id: "node",
    label: `Node.js ${process.version}`,
    status: nodeMajor >= MIN_NODE ? "ok" : "missing",
    detail: nodeMajor >= MIN_NODE ? "runtime OK" : `Node.js ${MIN_NODE}+ required: https://nodejs.org`,
  });

  const web = checkWebBuild();
  checks.push({
    id: "web",
    label: "Web interface build",
    status: web.ok ? "ok" : "missing",
    detail: web.detail,
  });

  const py = await findPython();
  checks.push({
    id: "python",
    label: "Python 3.10–3.12",
    status: py ? "ok" : "missing",
    detail: py ? `${py.version} (${py.source})` : "no suitable Python found",
  });

  if (py) {
    const deps = await checkEngineDeps(py.cmd);
    checks.push({
      id: "engine",
      label: "Engine packages (torch, uvicorn, librosa)",
      status: deps.ok ? "ok" : "missing",
      detail: deps.detail,
    });
  } else {
    checks.push({
      id: "engine",
      label: "Engine packages (torch, uvicorn, librosa)",
      status: "missing",
      detail: "needs Python first",
    });
  }

  const ff = await checkFfmpeg();
  checks.push({
    id: "ffmpeg",
    label: "ffmpeg",
    status: ff.ok ? "ok" : "warn",
    detail: ff.ok ? ff.detail : `${ff.detail} — some audio formats may fail`,
  });

  const logsDir = path.join(getRepoRoot(), "logs");
  let models = 0;
  try {
    const walk = (dir: string) => {
      for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, e.name);
        if (e.isDirectory()) walk(full);
        else if (e.name.endsWith(".pth") && !e.name.startsWith("G_") && !e.name.startsWith("D_")) models++;
      }
    };
    if (exists(logsDir)) walk(logsDir);
  } catch {
    /* ignore */
  }
  checks.push({
    id: "models",
    label: "Voice models",
    status: "ok",
    detail: models > 0 ? `${models} model(s) in logs/` : "none yet — use the Download tab",
  });

  const ready = checks.every((c) => c.status === "ok" || c.id === "ffmpeg" || c.id === "models");
  const status: SetupStatus = {
    ready,
    checks,
    python: py ? py.cmd : null,
    checkedAt: new Date().toISOString(),
  };
  cached = { at: Date.now(), status };
  return status;
}

async function streamRun(
  job: Job,
  cmd: string,
  args: string[],
  opts: { shell?: boolean } = {},
): Promise<void> {
  appendLog(job, `$ ${cmd} ${args.join(" ")}`);
  await new Promise<void>((resolve, reject) => {
    const child = spawn(cmd, args, {
      cwd: getRepoRoot(),
      windowsHide: true,
      shell: opts.shell || false,
      env: { ...process.env, PYTHONIOENCODING: "utf-8", UV_HTTP_TIMEOUT: "300" },
    });
    child.stdout?.on("data", (d: Buffer) => {
      for (const line of d.toString().split("\n")) {
        if (line.trim()) appendLog(job, line.trim().slice(0, 500));
      }
    });
    child.stderr?.on("data", (d: Buffer) => {
      for (const line of d.toString().split("\n")) {
        if (line.trim()) appendLog(job, line.trim().slice(0, 500));
      }
    });
    child.on("error", (e) => reject(new Error(`Failed to start ${cmd}: ${e.message}`)));
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} exited with code ${code}`));
    });
  });
}

async function bootstrapSystemPython(job: Job): Promise<string[]> {
  appendLog(job, "No system Python found — bootstrapping one…");
  if (process.platform === "win32") {
    appendLog(job, "Installing Python 3.12 via winget (no admin needed)…");
    await streamRun(job, "winget", [
      "install",
      "-e",
      "--id",
      "Python.Python.3.12",
      "--silent",
      "--accept-package-agreements",
      "--accept-source-agreements",
    ]);
    const retry = await findPython();
    if (!retry) {
      throw new Error(
        "winget install finished but no Python was found. Install Python 3.12 from https://www.python.org/downloads/ and press Install again.",
      );
    }
    return retry.cmd;
  }
  if (process.platform === "darwin") {
    const brew = await runCmd("brew", ["--version"], { timeoutMs: 15000 });
    if (brew.code !== 0) {
      throw new Error(
        "Install Homebrew (https://brew.sh) or Python 3.12 from python.org, then press Install again.",
      );
    }
    await streamRun(job, "brew", ["install", "python@3.12"]);
    const retry = await findPython();
    if (!retry) throw new Error("brew install finished but no Python was found.");
    return retry.cmd;
  }
  throw new Error(
    "Install Python 3.10–3.12 (e.g. sudo apt install python3-venv python3-pip), then press Install again.",
  );
}

export function startInstall(): Job {
  if (activeInstallId) {
    const existing = getJob(activeInstallId);
    if (existing && (existing.status === "queued" || existing.status === "running")) return existing;
  }
  const job = createJob("other", { setup: true });
  activeInstallId = job.id;
  void (async () => {
    setRunning(job);
    try {
      const root = getRepoRoot();
      const npmCmd = process.platform === "win32" ? "npm.cmd" : "npm";
      const npmShell = process.platform === "win32";

      const found = await findPython();
      const sysPy: string[] = found ? found.cmd : await bootstrapSystemPython(job);

      const venvPy =
        process.platform === "win32"
          ? path.join(root, ".venv", "Scripts", "python.exe")
          : path.join(root, ".venv", "bin", "python");
      if (!exists(venvPy)) {
        appendLog(job, "Creating app virtualenv (.venv)…");
        await streamRun(job, sysPy[0], [...sysPy.slice(1), "-m", "venv", path.join(root, ".venv")]);
      } else {
        appendLog(job, "App virtualenv already exists ✓");
      }

      appendLog(job, "Installing engine packages (torch + requirements — this takes a while)…");
      await streamRun(job, venvPy, ["-m", "pip", "install", "-U", "pip"]);
      const hasUv = (await runCmd("uv", ["--version"], { timeoutMs: 15000 })).code === 0;
      const torchIndex =
        process.platform === "darwin" ? [] : ["--extra-index-url", "https://download.pytorch.org/whl/cu128"];
      const reqFile = path.join(root, "requirements.txt");
      if (hasUv) {
        appendLog(job, "Using uv (fast installer)…");
        await streamRun(job, "uv", [
          "pip",
          "install",
          "--python",
          venvPy,
          "torch",
          ...torchIndex,
          // The torch wheel index also mirrors a few PyPI packages at older
          // versions; without this uv pins them to that index and resolution fails.
          ...(torchIndex.length > 0 ? ["--index-strategy", "unsafe-best-match"] : []),
          "-r",
          reqFile,
        ]);
      } else {
        await streamRun(job, venvPy, ["-m", "pip", "install", "torch", ...torchIndex]);
        await streamRun(job, venvPy, ["-m", "pip", "install", "-r", reqFile]);
      }

      process.env.PYTHON_BIN = venvPy;
      appendLog(job, `Using Python env: ${venvPy}`);

      appendLog(job, "Downloading base voice models and prerequisites (hubert, rmvpe)…");
      try {
        await streamRun(job, venvPy, ["core.py", "prerequisites", "--models", "--exe"]);
      } catch (e) {
        appendLog(job, `Note: Prerequisites download step: ${e}`);
      }

      if (exists(path.join(root, "app", "api", "package.json"))) {
        appendLog(job, "Installing web dependencies…");
        await streamRun(job, npmCmd, ["install", "--workspaces", "--include-workspace-root"], {
          shell: npmShell,
        });
        if (!exists(path.join(root, "app", "web", ".next", "standalone", "server.js"))) {
          if (await webDevServerRunning()) {
            appendLog(
              job,
              `! Skipping web build — a dev server is already serving port ${WEB_PORT}. ` +
                "Dev mode does not need the production bundle; to build it, stop `npm run dev` and run `npm run build`.",
            );
          } else {
            appendLog(job, "Building web interface…");
            try {
              await streamRun(job, npmCmd, ["run", "build"], { shell: npmShell });
            } catch (e) {
              appendLog(job, `! Web build failed (${e}) — everything else installed; run \`npm run build\` manually.`);
            }
          }
        }
      } else {
        appendLog(job, "Packaged app — web bundles already included ✓");
      }

      cached = null;
      const final = await getStatus(true);
      for (const c of final.checks) {
        appendLog(
          job,
          `${c.status === "ok" ? "✓" : c.status === "warn" ? "!" : "✗"} ${c.label}: ${c.detail}`,
        );
      }
      if (!final.ready) throw new Error("Setup finished but some required checks still fail — see above.");
      setDone(job, { message: "Setup complete — Applio is ready.", ready: true });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      appendLog(job, `ERROR: ${message}`);
      setError(job, message);
    } finally {
      activeInstallId = null;
    }
  })();
  return job;
}

export function startPrerequisites(py: string[] | null): Job {
  const job = createJob("other", { setup: "prerequisites" });
  void (async () => {
    setRunning(job);
    try {
      const exe = py
        ? py[0]
        : process.env.PYTHON_BIN || (process.platform === "win32" ? "python" : "python3");
      const prefix = py ? py.slice(1) : [];
      await streamRun(job, exe, [
        ...prefix,
        "core.py",
        "prerequisites",
        "--pretraineds-hifigan",
        "--models",
        "--exe",
      ]);
      setDone(job, { message: "Engine models downloaded." });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      appendLog(job, `ERROR: ${message}`);
      setError(job, message);
    }
  })();
  return job;
}

export function venvPythonPath(): string {
  const root = getRepoRoot();
  return process.platform === "win32"
    ? path.join(root, ".venv", "Scripts", "python.exe")
    : path.join(root, ".venv", "bin", "python");
}
