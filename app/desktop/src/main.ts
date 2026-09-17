// Applio Electron shell (dev + packaged production).

import { type ChildProcess, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { app, BrowserWindow, clipboard, dialog, ipcMain, shell } from "electron";

const isDev: boolean = !app.isPackaged;
const API_PORT: string = process.env.API_PORT || "8000";
const WEB_PORT: string = process.env.WEB_PORT || "3000";
const WEB_URL: string = process.env.WEB_URL || `http://127.0.0.1:${WEB_PORT}/`;

let apiProc: ChildProcess | null = null;
let webProc: ChildProcess | null = null;
let mainWindow: BrowserWindow | null = null;
let splash: BrowserWindow | null = null;

function repoRoot(): string {
  if (process.env.APPLIO_ROOT && fs.existsSync(process.env.APPLIO_ROOT)) {
    return path.resolve(process.env.APPLIO_ROOT);
  }
  if (isDev) return path.resolve(__dirname, "..", "..", "..");
  return path.resolve(__dirname, "..");
}

function startProdBackends(): void {
  const root = repoRoot();
  const serverEntry = path.join(root, "app", "api", "dist", "index.js");
  const nextStandalone = path.join(root, "app", "web", ".next", "standalone", "server.js");
  const env: NodeJS.ProcessEnv = {
    ...process.env,
    API_PORT,
    APPLIO_ROOT: root,
    PORT: WEB_PORT,
    HOSTNAME: "127.0.0.1",
  };
  if (fs.existsSync(serverEntry)) {
    apiProc = spawn(process.execPath, [serverEntry], { env, windowsHide: true });
    pipeProcOutput(apiProc, "api");
  } else {
    console.warn("[electron] api bundle missing:", serverEntry);
  }
  if (fs.existsSync(nextStandalone)) {
    webProc = spawn(process.execPath, [nextStandalone], { env, windowsHide: true });
    pipeProcOutput(webProc, "web");
  } else {
    console.warn("[electron] web standalone missing:", nextStandalone);
  }
}

function stopBackends(): void {
  for (const p of [apiProc, webProc]) {
    try {
      p?.kill();
    } catch {
      /* already gone */
    }
  }
  apiProc = null;
  webProc = null;
}

// Backend output goes to rotating log files AND an in-memory tail so the
// failure dialog can show paths, open the folder and copy diagnostics.
const bootLogTails: string[] = [];

function launcherLogDir(): string {
  // getPath('logs') throws unless setAppLogsPath() ran first: fall back to
  // userData so logging can never break the boot sequence.
  try {
    const dir = path.join(app.getPath("logs"), "applio");
    fs.mkdirSync(dir, { recursive: true });
    return dir;
  } catch {
    const dir = path.join(app.getPath("userData"), "logs", "applio");
    fs.mkdirSync(dir, { recursive: true });
    return dir;
  }
}

function pipeProcOutput(proc: ChildProcess, tag: "api" | "web"): void {
  const file = path.join(launcherLogDir(), `${tag}.log`);
  const push = (line: string) => {
    const entry = `[${tag}] ${line}`;
    bootLogTails.push(entry);
    if (bootLogTails.length > 200) bootLogTails.splice(0, bootLogTails.length - 200);
    try {
      fs.appendFileSync(file, `${new Date().toISOString()} ${line}\n`);
    } catch {
      /* disk full / locked: console still has it */
    }
  };
  proc.stdout?.on("data", (d: Buffer) => {
    const text = d.toString();
    console.log(`[${tag}]`, text);
    for (const line of text.split("\n")) if (line.trim()) push(line.trim().slice(0, 1000));
  });
  proc.stderr?.on("data", (d: Buffer) => {
    const text = d.toString();
    console.error(`[${tag}]`, text);
    for (const line of text.split("\n")) if (line.trim()) push(`STDERR ${line.trim().slice(0, 1000)}`);
  });
}

function venvPython(): string {
  const root = repoRoot();
  const win = path.join(root, ".venv", "Scripts", "python.exe");
  const nix = path.join(root, ".venv", "bin", "python");
  if (process.platform === "win32") return win;
  return nix;
}

function diagnosticsReport(): string {
  const venv = venvPython();
  return [
    `Applio ${app.getVersion()} · ${process.platform} ${process.arch}`,
    `Electron ${process.versions.electron} · Node ${process.versions.node}`,
    `Python env: ${venv} (${fs.existsSync(venv) ? "found" : "MISSING"})`,
    `Ports: API=${API_PORT} WEB=${WEB_PORT}`,
    `Logs: ${launcherLogDir()}`,
    `--- backend tail ---`,
    ...bootLogTails.slice(-15),
  ].join("\n");
}

async function reportBootFailure(): Promise<"retry" | "quit"> {
  const venv = venvPython();
  const detail = [
    "The bundled engine did not come up within 60s, so there is nothing to show yet.",
    "",
    "Checklist:",
    `• Python env: ${venv} (${fs.existsSync(venv) ? "found" : "MISSING — run setup first"})`,
    `• Ports ${API_PORT}/${WEB_PORT} free (no other Applio running?)`,
    `• Logs: ${launcherLogDir()}`,
    "",
    "Recent backend output:",
    ...bootLogTails.slice(-8).map((l) => `• ${l}`),
  ].join("\n");
  for (;;) {
    const { response } = await dialog.showMessageBox({
      type: "error",
      title: "Applio failed to start",
      message: "Applio failed to start",
      detail,
      buttons: ["Retry", "Copy diagnostics", "Open log folder", "Quit"],
      defaultId: 0,
      cancelId: 3,
      noLink: true,
    });
    if (response === 0) return "retry";
    if (response === 1) {
      clipboard.writeText(diagnosticsReport());
      continue;
    }
    if (response === 2) {
      void shell.openPath(launcherLogDir());
      continue;
    }
    return "quit";
  }
}

async function waitFor(url: string, tries = 60, onTick?: (n: number) => void): Promise<boolean> {
  for (let i = 0; i < tries; i++) {
    try {
      const r = await fetch(url);
      if (r.ok) return true;
    } catch {
      /* not up yet */
    }
    onTick?.(i + 1);
    await new Promise((r) => setTimeout(r, 1000));
  }
  return false;
}

const SPLASH_HTML = `data:text/html,${encodeURIComponent(`<!doctype html>
<html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:#0a0a0a;color:#e7e5e4;font-family:system-ui,sans-serif;display:flex;align-items:center;justify-content:center}
.wrap{text-align:center}
.brand{font-size:28px;font-weight:800;letter-spacing:-0.02em}
.brand small{font-size:11px;color:#a3a3a3;font-weight:500;margin-left:8px;letter-spacing:0.08em}
.spin{width:28px;height:28px;margin:22px auto 14px;border-radius:50%;border:3px solid rgba(255,255,255,.15);border-top-color:#fff;animation:sp 0.9s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
#st{font-size:13px;color:#a3a3a3;min-height:20px}
</style></head><body><div class="wrap">
<div class="brand">Applio<small>STUDIO</small></div>
<div class="spin"></div>
<div id="st">Starting…</div>
</div></body></html>`)}`;

function showSplash(): void {
  if (splash && !splash.isDestroyed()) return;
  splash = new BrowserWindow({
    width: 420,
    height: 320,
    resizable: false,
    minimizable: false,
    maximizable: false,
    frame: false,
    center: true,
    show: true,
    backgroundColor: "#0a0a0a",
    webPreferences: { contextIsolation: true, nodeIntegration: false },
  });
  void splash.loadURL(SPLASH_HTML);
  splash.on("closed", () => {
    splash = null;
  });
}

function setSplashStatus(text: string): void {
  if (!splash || splash.isDestroyed()) return;
  const esc = text.replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\n/g, " ");
  void splash.webContents
    .executeJavaScript(`(function(){var el=document.getElementById('st');if(el)el.textContent='${esc}';})()`)
    .catch(() => {});
}

function closeSplash(): void {
  try {
    splash?.close();
  } catch {
    /* already gone */
  }
  splash = null;
}

interface SavedWindowState {
  width: number;
  height: number;
  x?: number;
  y?: number;
}

function windowStatePath(): string {
  return path.join(app.getPath("userData"), "window-state.json");
}

function loadWindowState(): SavedWindowState | null {
  try {
    const raw = JSON.parse(fs.readFileSync(windowStatePath(), "utf-8")) as Partial<SavedWindowState>;
    if (
      typeof raw.width !== "number" ||
      typeof raw.height !== "number" ||
      raw.width < 960 ||
      raw.height < 640 ||
      raw.width > 7680 ||
      raw.height > 4320
    ) {
      return null;
    }
    const state: SavedWindowState = { width: Math.round(raw.width), height: Math.round(raw.height) };
    if (typeof raw.x === "number" && typeof raw.y === "number") {
      state.x = Math.round(raw.x);
      state.y = Math.round(raw.y);
    }
    return state;
  } catch {
    return null;
  }
}

function saveWindowState(): void {
  try {
    if (!mainWindow || mainWindow.isDestroyed()) return;
    const [width, height] = mainWindow.getSize();
    const [x, y] = mainWindow.getPosition();
    const state: SavedWindowState = { width, height, x, y };
    fs.writeFileSync(windowStatePath(), JSON.stringify(state));
  } catch {
    /* non-fatal */
  }
}

async function createWindow(): Promise<void> {
  if (!isDev) {
    showSplash();
    let attempt = 0;
    for (;;) {
      attempt += 1;
      stopBackends();
      setSplashStatus(attempt > 1 ? `Retrying… (attempt ${attempt})` : "Starting engine…");
      startProdBackends();
      const ok = await waitFor(`http://127.0.0.1:${WEB_PORT}/`, 60, (n) =>
        setSplashStatus(`Waiting for studio… (${n}/60)`),
      );
      if (ok) break;
      const action = await reportBootFailure();
      if (action === "retry") continue;
      closeSplash();
      stopBackends();
      app.quit();
      return;
    }
    closeSplash();
  }

  const iconPath = path.join(repoRoot(), "assets", "ICON.ico");
  const saved = loadWindowState();
  mainWindow = new BrowserWindow({
    width: saved?.width ?? 1280,
    height: saved?.height ?? 860,
    x: saved?.x,
    y: saved?.y,
    minWidth: 960,
    minHeight: 640,
    title: "Applio",
    frame: false,
    show: false,
    center: saved === null,
    backgroundColor: "#0a0a0a",
    autoHideMenuBar: true,
    ...(fs.existsSync(iconPath) ? { icon: iconPath } : {}),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // Paint only when the first frame is ready instead of grabbing focus.
  mainWindow.once("ready-to-show", () => {
    mainWindow?.show();
    mainWindow?.focus();
  });

  const target = isDev ? WEB_URL : `http://127.0.0.1:${WEB_PORT}/`;
  console.log("[electron] loading target:", target);
  void mainWindow.loadURL(target);

  // Notify the renderer of maximize state so its chrome never drifts
  // (Win+Arrow, snap layouts and the OS window menu bypass our IPC toggle).
  mainWindow.on("maximize", () => {
    mainWindow?.webContents.send("window:maximize-changed", true);
  });
  mainWindow.on("unmaximize", () => {
    mainWindow?.webContents.send("window:maximize-changed", false);
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
  mainWindow.on("close", () => {
    saveWindowState();
  });
}

// Window control handlers
ipcMain.on("window:minimize", (event) => {
  const win = BrowserWindow.fromWebContents(event.sender) || mainWindow;
  win?.minimize();
});

ipcMain.on("window:toggle-maximize", (event) => {
  const win = BrowserWindow.fromWebContents(event.sender) || mainWindow;
  if (!win) return;
  if (win.isMaximized()) {
    win.unmaximize();
  } else {
    win.maximize();
  }
});

ipcMain.on("window:close", (event) => {
  const win = BrowserWindow.fromWebContents(event.sender) || mainWindow;
  win?.close();
});

ipcMain.handle("window:is-maximized", (event) => {
  const win = BrowserWindow.fromWebContents(event.sender) || mainWindow;
  return win?.isMaximized() ?? false;
});

// One studio at a time: a second launch focuses the running window instead
// of fighting over ports with a duplicate backend stack.
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
  // Required before any getPath('logs') call (throws otherwise).
  try {
    app.setAppLogsPath();
  } catch {
    /* launcherLogDir falls back to userData */
  }
  app.whenReady().then(createWindow);
}

app.on("window-all-closed", () => {
  apiProc?.kill();
  webProc?.kill();
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) void createWindow();
});

app.on("before-quit", () => {
  apiProc?.kill();
  webProc?.kill();
});
