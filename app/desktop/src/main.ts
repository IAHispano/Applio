// Applio Electron shell (dev + packaged production).

import { type ChildProcess, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { app, BrowserWindow, dialog, ipcMain } from "electron";

const isDev: boolean = !app.isPackaged;
const API_PORT: string = process.env.API_PORT || "8000";
const WEB_PORT: string = process.env.WEB_PORT || "3000";
const WEB_URL: string = process.env.WEB_URL || `http://127.0.0.1:${WEB_PORT}/`;

let apiProc: ChildProcess | null = null;
let webProc: ChildProcess | null = null;

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
    apiProc.stdout?.on("data", (d: Buffer) => console.log("[api]", d.toString()));
    apiProc.stderr?.on("data", (d: Buffer) => console.error("[api]", d.toString()));
  } else {
    console.warn("[electron] api bundle missing:", serverEntry);
  }
  if (fs.existsSync(nextStandalone)) {
    webProc = spawn(process.execPath, [nextStandalone], { env, windowsHide: true });
    webProc.stdout?.on("data", (d: Buffer) => console.log("[web]", d.toString()));
    webProc.stderr?.on("data", (d: Buffer) => console.error("[web]", d.toString()));
  } else {
    console.warn("[electron] web standalone missing:", nextStandalone);
  }
  // The Windows installer is per-user, so APPLIO_ROOT stays writable for setup.
}

async function waitFor(url: string, tries = 60): Promise<boolean> {
  for (let i = 0; i < tries; i++) {
    try {
      const r = await fetch(url);
      if (r.ok) return true;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  return false;
}

async function createWindow(): Promise<void> {
  if (!isDev) startProdBackends();
  else await waitFor(`http://127.0.0.1:${API_PORT}/api/health`, 10).catch(() => false);

  const iconPath = path.join(repoRoot(), "assets", "ICON.ico");
  const win = new BrowserWindow({
    width: 1280,
    height: 860,
    title: "Applio",
    frame: false,
    ...(fs.existsSync(iconPath) ? { icon: iconPath } : {}),
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
    },
  });

  ipcMain.on("window:minimize", () => win.minimize());
  ipcMain.on("window:toggle-maximize", () => {
    if (win.isMaximized()) win.unmaximize();
    else win.maximize();
  });
  ipcMain.on("window:close", () => win.close());
  const target = isDev ? WEB_URL : `http://127.0.0.1:${WEB_PORT}/`;
  if (!isDev) {
    const ok = await waitFor(target, 60);
    if (!ok) {
      dialog.showErrorBox(
        "Applio failed to start",
        "Bundled web/API did not come up. Check the setup screen requirements and try again.",
      );
    }
  }
  await win.loadURL(target);
}

app.whenReady().then(createWindow);
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
