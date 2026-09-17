"use client";

import { ChevronLeft, ChevronRight, Maximize2, Minimize2, Minus, RefreshCcw, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

interface WindowControls {
  minimize: () => void;
  toggleMaximize: () => void;
  close: () => void;
}

export default function TitleBar() {
  const router = useRouter();
  const [maximized, setMaximized] = useState(false);
  const [_controls, setControls] = useState<WindowControls | null>(null);

  useEffect(() => {
    const bridge = (window as unknown as { applio?: { controls?: WindowControls } }).applio;
    if (bridge?.controls) {
      setControls(bridge.controls);
    }
  }, []);

  function handleMinimize() {
    const bridge = (window as unknown as { applio?: { controls?: WindowControls } }).applio;
    if (bridge?.controls) {
      bridge.controls.minimize();
    }
  }

  function handleMaximize() {
    const bridge = (window as unknown as { applio?: { controls?: WindowControls } }).applio;
    if (bridge?.controls) {
      bridge.controls.toggleMaximize();
      setMaximized((v) => !v);
    } else {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
        setMaximized(true);
      } else {
        document.exitFullscreen().catch(() => {});
        setMaximized(false);
      }
    }
  }

  function handleClose() {
    const bridge = (window as unknown as { applio?: { controls?: WindowControls } }).applio;
    if (bridge?.controls) {
      bridge.controls.close();
    } else {
      window.close();
    }
  }

  function goBack() {
    if (typeof window !== "undefined" && window.history.length > 1) {
      router.back();
    } else {
      router.push("/");
    }
  }

  function goForward() {
    router.forward();
  }

  return (
    <header className="h-9 w-full select-none bg-[#0c0c0c] border-b border-white/5 flex items-center justify-between px-3 shrink-0 [-webkit-app-region:drag] z-50">
      {/* Left: Navigation actions */}
      <div className="flex items-center gap-1 shrink-0 [-webkit-app-region:no-drag]">
        <button type="button" className="titlebar-btn" onClick={goBack} title="Back" aria-label="Back">
          <ChevronLeft className="w-4 h-4 text-neutral-300" />
        </button>
        <button
          type="button"
          className="titlebar-btn"
          onClick={goForward}
          title="Forward"
          aria-label="Forward"
        >
          <ChevronRight className="w-4 h-4 text-neutral-300" />
        </button>
        <button
          type="button"
          className="titlebar-btn"
          onClick={() => window.location.reload()}
          title="Reload"
          aria-label="Reload"
        >
          <RefreshCcw className="w-3.5 h-3.5 text-neutral-300" />
        </button>
      </div>

      {/* Center: Draggable App Title */}
      <div className="flex-1 flex items-center justify-center pointer-events-none">
        <span className="text-xs font-medium text-neutral-400 tracking-wider flex items-center gap-2">
          <span className="text-neutral-300">Applio</span>
          <span className="text-[10px] text-neutral-500 font-normal">v3.6</span>
        </span>
      </div>

      {/* Right: Window Controls (Always visible) */}
      <div className="flex items-center gap-1 shrink-0 [-webkit-app-region:no-drag] justify-end">
        <button
          type="button"
          className="titlebar-btn"
          onClick={handleMinimize}
          title="Minimize"
          aria-label="Minimize"
        >
          <Minus className="w-3.5 h-3.5 text-neutral-300" />
        </button>
        <button
          type="button"
          className="titlebar-btn"
          onClick={handleMaximize}
          title={maximized ? "Restore" : "Maximize"}
          aria-label={maximized ? "Restore" : "Maximize"}
        >
          {maximized ? (
            <Minimize2 className="w-3.5 h-3.5 text-neutral-300" />
          ) : (
            <Maximize2 className="w-3.5 h-3.5 text-neutral-300" />
          )}
        </button>
        <button
          type="button"
          className="titlebar-btn titlebar-btn-close"
          onClick={handleClose}
          title="Close"
          aria-label="Close"
        >
          <X className="w-3.5 h-3.5 text-neutral-300" />
        </button>
      </div>
    </header>
  );
}
