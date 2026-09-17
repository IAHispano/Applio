import { contextBridge, ipcRenderer } from "electron";

export interface WindowControls {
  minimize: () => void;
  toggleMaximize: () => void;
  close: () => void;
}

export interface ApplioBridge {
  platform: NodeJS.Platform;
  versions: NodeJS.ProcessVersions;
  controls: WindowControls;
}

declare global {
  interface Window {
    applio: ApplioBridge;
  }
}

contextBridge.exposeInMainWorld("applio", {
  platform: process.platform,
  versions: process.versions,
  controls: {
    minimize: () => ipcRenderer.send("window:minimize"),
    toggleMaximize: () => ipcRenderer.send("window:toggle-maximize"),
    close: () => ipcRenderer.send("window:close"),
  },
} satisfies ApplioBridge);
