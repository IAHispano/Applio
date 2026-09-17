import { type ChildProcess, spawn } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import { errMsg } from "../errors";
import { getPythonBin, getRepoRoot } from "../python";

const router = Router();
const TB_PORT = Number(process.env.TB_PORT || 6007);

let tbProc: ChildProcess | null = null;
let tbStartedAt: string | null = null;
let startingPromise: Promise<{ ok: boolean; url: string; error?: string }> | null = null;

function tbUrl(): string {
  return `http://127.0.0.1:${TB_PORT}`;
}

function portOpen(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (result: boolean) => {
      if (settled) return;
      settled = true;
      try {
        s.destroy();
      } catch {
        /* noop */
      }
      resolve(result);
    };
    const s = net.connect({ port, host: "127.0.0.1" });
    s.once("connect", () => finish(true));
    s.on("error", () => finish(false));
    const timer = setTimeout(() => finish(false), 1500);
    timer.unref?.();
  });
}

export function stopTensorboard(): void {
  try {
    tbProc?.kill();
  } catch {
    /* already stopped */
  }
  tbProc = null;
  tbStartedAt = null;
  startingPromise = null;
}

export async function startTensorboard(): Promise<{ ok: boolean; url: string; error?: string }> {
  const isReachable = await portOpen(TB_PORT);
  if (tbProc && tbProc.exitCode === null && isReachable) {
    return { ok: true, url: tbUrl() };
  }

  if (startingPromise) {
    return startingPromise;
  }

  startingPromise = (async () => {
    try {
      stopTensorboard();

      const root = getRepoRoot();
      const logsDir = path.join(root, "logs");
      if (!fs.existsSync(logsDir)) {
        fs.mkdirSync(logsDir, { recursive: true });
      }

      const pythonBin = getPythonBin();
      console.log(`[tensorboard] starting on ${tbUrl()} using ${pythonBin}`);

      tbProc = spawn(
        pythonBin,
        ["-m", "tensorboard.main", "--logdir", "logs", "--host", "127.0.0.1", "--port", String(TB_PORT)],
        {
          cwd: root,
          windowsHide: true,
          env: { ...process.env, PYTHONIOENCODING: "utf-8" },
        },
      );

      tbStartedAt = new Date().toISOString();

      tbProc.on("error", (err) => {
        console.error("[tensorboard] process error:", err);
        tbProc = null;
      });

      tbProc.on("exit", (code) => {
        if (code !== 0 && code !== null) {
          console.warn(`[tensorboard] exited with code ${code}`);
        }
        tbProc = null;
      });

      for (let i = 0; i < 20; i++) {
        await new Promise((r) => setTimeout(r, 1000));
        if (tbProc?.exitCode !== null && tbProc?.exitCode !== undefined) {
          const err = "TensorBoard process exited immediately. Check logs or pip install tensorboard.";
          console.error(`[tensorboard] ${err}`);
          tbProc = null;
          return { ok: false, url: tbUrl(), error: err };
        }
        if (await portOpen(TB_PORT)) {
          console.log(`[tensorboard] ready on ${tbUrl()}`);
          return { ok: true, url: tbUrl() };
        }
      }

      return { ok: false, url: tbUrl(), error: "TensorBoard did not come up in time." };
    } catch (err) {
      stopTensorboard();
      return { ok: false, url: tbUrl(), error: errMsg(err) || "Could not start TensorBoard" };
    } finally {
      startingPromise = null;
    }
  })();

  return startingPromise;
}

export function autoStartTensorboard(): void {
  // Asynchronously launch in the background at startup without blocking
  void startTensorboard().catch((err) => {
    console.warn("[tensorboard] auto-start failed:", err);
  });
}

router.get("/status", async (_req: Request, res: Response) => {
  const alive = tbProc !== null && tbProc.exitCode === null;
  const reachable = await portOpen(TB_PORT);
  const isRunning = alive && reachable;

  // Auto-start in background if not already running or starting
  if (!isRunning && !startingPromise) {
    autoStartTensorboard();
  }

  res.json({
    running: isRunning,
    starting: startingPromise !== null,
    url: tbUrl(),
    startedAt: tbStartedAt,
  });
});

router.post("/start", async (_req: Request, res: Response) => {
  const result = await startTensorboard();
  if (result.ok) {
    return res.json({ ok: true, url: result.url, startedAt: tbStartedAt });
  }
  return res.status(500).json({ error: result.error || "Could not start TensorBoard" });
});

router.post("/stop", (_req: Request, res: Response) => {
  stopTensorboard();
  res.json({ ok: true });
});

export default router;
