import { contextBridge, ipcRenderer } from "electron";

export interface WindowControls {
  minimize: () => void;
  toggleMaximize: () => void;
  close: () => void;
  isMaximized: () => Promise<boolean>;
  onMaximizeChanged: (cb: (maximized: boolean) => void) => () => void;
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
    isMaximized: () => ipcRenderer.invoke("window:is-maximized"),
    onMaximizeChanged: (cb: (maximized: boolean) => void) => {
      const listener = (_event: unknown, value: unknown) => cb(value === true);
      ipcRenderer.on("window:maximize-changed", listener);
      return () => ipcRenderer.removeListener("window:maximize-changed", listener);
    },
  },
} satisfies ApplioBridge);
