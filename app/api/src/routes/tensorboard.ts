import { type ChildProcess, spawn } from "node:child_process";
import net from "node:net";
import { type Request, type Response, Router } from "express";
import { errMsg } from "../errors";

const router = Router();
const TB_PORT = Number(process.env.TB_PORT || 6007);

let tbProc: ChildProcess | null = null;
let tbStartedAt: string | null = null;

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

router.get("/status", async (_req: Request, res: Response) => {
  const alive = tbProc !== null && tbProc.exitCode === null;
  const reachable = await portOpen(TB_PORT);
  res.json({ running: alive && reachable, url: tbUrl(), startedAt: tbStartedAt });
});

router.post("/start", async (_req: Request, res: Response) => {
  try {
    if (tbProc && tbProc.exitCode === null && (await portOpen(TB_PORT))) {
      return res.json({ ok: true, url: tbUrl(), reused: true });
    }
    tbProc?.kill();
    tbProc = spawn("tensorboard", ["--logdir", "logs", "--host", "127.0.0.1", "--port", String(TB_PORT)], {
      cwd: process.env.APPLIO_ROOT,
      windowsHide: true,
    });
    tbStartedAt = new Date().toISOString();
    tbProc.on("error", () => {
      tbProc = null;
    });
    for (let i = 0; i < 20; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      if (tbProc?.exitCode !== null && tbProc?.exitCode !== undefined) {
        tbProc = null;
        return res
          .status(500)
          .json({ error: "TensorBoard exited immediately. Is it installed? (pip install tensorboard)" });
      }
      if (await portOpen(TB_PORT)) return res.json({ ok: true, url: tbUrl(), startedAt: tbStartedAt });
    }
    return res.status(504).json({ error: "TensorBoard did not come up in time." });
  } catch (err) {
    tbProc = null;
    return res.status(500).json({ error: errMsg(err) || "Could not start TensorBoard" });
  }
});

router.post("/stop", (_req: Request, res: Response) => {
  tbProc?.kill();
  tbProc = null;
  tbStartedAt = null;
  res.json({ ok: true });
});

export default router;
