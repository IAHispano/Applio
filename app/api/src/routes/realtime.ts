import { type ChildProcess, spawn } from "node:child_process";
import fs from "node:fs";
import type http from "node:http";
import net from "node:net";
import path from "node:path";
import { type Request, type Response, Router } from "express";
import { type RawData, WebSocket, WebSocketServer } from "ws";
import { errMsg } from "../errors";
import { getRepoRoot } from "../python";

const router = Router();
export const RT_PORT = Number(process.env.RT_PORT || 8001);

let rtProc: ChildProcess | null = null;
let rtStartedAt: string | null = null;
let rtLogs: string[] = [];

function backend(pathname: string): string {
  return `http://127.0.0.1:${RT_PORT}/api${pathname}`;
}

function portOpen(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const s = net.connect(port, "127.0.0.1");
    s.on("connect", () => {
      s.end();
      resolve(true);
    });
    s.on("error", () => resolve(false));
    setTimeout(() => {
      s.destroy();
      resolve(false);
    }, 1500).unref?.();
  });
}

router.get("/status", async (_req: Request, res: Response) => {
  const alive = rtProc !== null && rtProc.exitCode === null;
  const reachable = await portOpen(RT_PORT);
  res.json({
    running: alive && reachable,
    startedAt: rtStartedAt,
    wsAudio: `/api/realtime/ws-audio`,
    wsConfig: `/api/realtime/change-config`,
    logs: rtLogs.slice(-30),
  });
});

router.post("/start", async (_req: Request, res: Response) => {
  try {
    if (rtProc && rtProc.exitCode === null && (await portOpen(RT_PORT))) {
      return res.json({ ok: true, reused: true, startedAt: rtStartedAt });
    }
    rtProc?.kill();
    rtLogs = [];
    rtProc = spawn(
      "python",
      ["-m", "uvicorn", "rvc.realtime.client:app", "--host", "127.0.0.1", "--port", String(RT_PORT)],
      {
        cwd: getRepoRoot(),
        env: { ...process.env, PYTHONIOENCODING: "utf-8" },
        windowsHide: true,
      },
    );
    rtStartedAt = new Date().toISOString();
    rtProc.stdout?.on("data", (d: Buffer) => {
      rtLogs.push(d.toString().trim().slice(0, 500));
      if (rtLogs.length > 200) rtLogs = rtLogs.slice(-200);
    });
    rtProc.stderr?.on("data", (d: Buffer) => {
      rtLogs.push(`[stderr] ${d.toString().trim().slice(0, 500)}`);
      if (rtLogs.length > 200) rtLogs = rtLogs.slice(-200);
    });
    rtProc.on("error", (e) => {
      rtLogs.push(`spawn error: ${String(e)}`);
      rtProc = null;
    });
    for (let i = 0; i < 30; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      if (!rtProc || (rtProc.exitCode !== null && rtProc.exitCode !== undefined)) {
        rtProc = null;
        return res.status(500).json({
          error: "Realtime engine exited. Check uvicorn is installed and a GPU/model is available.",
          logs: rtLogs.slice(-10),
        });
      }
      if (await portOpen(RT_PORT)) return res.json({ ok: true, startedAt: rtStartedAt });
    }
    return res
      .status(504)
      .json({ error: "Realtime engine did not come up in time.", logs: rtLogs.slice(-10) });
  } catch (err) {
    return res.status(500).json({ error: errMsg(err) });
  }
});

router.post("/stop", (_req: Request, res: Response) => {
  rtProc?.kill();
  rtProc = null;
  rtStartedAt = null;
  res.json({ ok: true });
});

router.post("/record", async (req: Request, res: Response) => {
  try {
    const r = await fetch(backend("/record"), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(req.body || {}),
    });
    const body = await r.json().catch(() => ({}));
    res.status(r.status).json(body);
  } catch (err) {
    res.status(502).json({ error: `Engine unreachable: ${errMsg(err)}` });
  }
});

function cfgPath(): string {
  return path.join(getRepoRoot(), "assets", "config.json");
}
router.get("/config", (_req: Request, res: Response) => {
  try {
    const cfg = (fs.existsSync(cfgPath()) ? JSON.parse(fs.readFileSync(cfgPath(), "utf-8")) : {}) as {
      realtime?: unknown;
    };
    res.json({ realtime: cfg.realtime || {} });
  } catch (err) {
    res.status(500).json({ error: errMsg(err) });
  }
});
router.put("/config", (req: Request, res: Response) => {
  try {
    const cfg = (fs.existsSync(cfgPath()) ? JSON.parse(fs.readFileSync(cfgPath(), "utf-8")) : {}) as {
      realtime?: Record<string, unknown>;
    };
    cfg.realtime = { ...(cfg.realtime || {}), ...((req.body || {}) as Record<string, unknown>) };
    fs.writeFileSync(cfgPath(), JSON.stringify(cfg, null, 2));
    res.json({ ok: true, realtime: cfg.realtime });
  } catch (err) {
    res.status(500).json({ error: errMsg(err) });
  }
});

/** Attach WS upgrade proxying: /api/realtime/ws-audio + /change-config -> engine (binary-safe). */
export function attachRealtimeProxy(server: http.Server) {
  const wss = new WebSocketServer({ noServer: true });
  server.on("upgrade", (req, socket, head) => {
    const url = req.url || "";
    let target: string | null = null;
    if (url.startsWith("/api/realtime/ws-audio")) target = `ws://127.0.0.1:${RT_PORT}/api/ws-audio`;
    else if (url.startsWith("/api/realtime/change-config"))
      target = `ws://127.0.0.1:${RT_PORT}/api/change-config`;
    if (!target) return; // not ours
    wss.handleUpgrade(req, socket, head, (client) => proxySocket(client, target as string));
  });
}

function proxySocket(client: WebSocket, target: string) {
  const upstream = new WebSocket(target);
  const queue: Array<{ data: RawData; binary: boolean }> = [];
  client.on("message", (data, isBinary) => {
    if (upstream.readyState === WebSocket.OPEN) upstream.send(data, { binary: isBinary });
    else queue.push({ data, binary: isBinary });
  });
  upstream.on("open", () => {
    for (const m of queue) {
      if (upstream.readyState === WebSocket.OPEN) upstream.send(m.data, { binary: m.binary });
    }
    queue.length = 0;
  });
  upstream.on("message", (data, isBinary) => {
    if (client.readyState === WebSocket.OPEN) client.send(data, { binary: isBinary });
  });
  const close = () => {
    try {
      client.close();
    } catch {
      /* noop */
    }
    try {
      upstream.close();
    } catch {
      /* noop */
    }
  };
  client.on("close", close);
  upstream.on("close", close);
  client.on("error", close);
  upstream.on("error", close);
}

export default router;
