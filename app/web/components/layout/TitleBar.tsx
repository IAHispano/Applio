"use client";

import { ChevronLeft, ChevronRight, Maximize2, Minimize2, Minus, RefreshCcw, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { useI18n } from "../../lib/i18n";

interface WindowControls {
  minimize: () => void;
  toggleMaximize: () => void;
  close: () => void;
  isMaximized: () => Promise<boolean>;
  onMaximizeChanged: (cb: (maximized: boolean) => void) => () => void;
}

function getControls(): WindowControls | null {
  if (typeof window === "undefined") return null;
  const bridge = (window as unknown as { applio?: { controls?: WindowControls } }).applio;
  return bridge?.controls ?? null;
}

export default function TitleBar() {
  const router = useRouter();
  const { t } = useI18n();
  const [maximized, setMaximized] = useState(false);
  // No shell bridge (plain browser) means no OS window to control: the
  // minimize/close buttons would lie, so they are hidden there. Maximize
  // stays as an honest fullscreen toggle with matching labels.
  const [hasBridge, setHasBridge] = useState(false);

  // Source of truth lives in the shell: read once on mount and follow its
  // maximize-changed events (snap layouts, Win+Arrow and the OS menu bypass
  // our toggle). No shell (plain browser) keeps the previous fullscreen stand-in.
  useEffect(() => {
    const controls = getControls();
    setHasBridge(controls !== null);
    if (!controls) return;
    let live = true;
    controls
      .isMaximized()
      .then((v) => {
        if (live) setMaximized(v);
      })
      .catch(() => {});
    const unsubscribe = controls.onMaximizeChanged((v) => {
      if (live) setMaximized(v);
    });
    return () => {
      live = false;
      unsubscribe();
    };
  }, []);

  function handleMinimize() {
    getControls()?.minimize();
  }

  function handleMaximize() {
    const controls = getControls();
    if (controls) {
      controls.toggleMaximize();
      return;
    }
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
      setMaximized(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setMaximized(false);
    }
  }

  function handleClose() {
    const controls = getControls();
    if (controls) {
      controls.close();
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

  // Desktop convention: double-click on the drag region toggles maximize.
  // Clicks originating from buttons keep their own behavior.
  function handleHeaderDoubleClick(e: React.MouseEvent) {
    if ((e.target as HTMLElement).closest("button")) return;
    handleMaximize();
  }

  return (
    <header
      className="flex h-9 w-full shrink-0 select-none items-center justify-between border-b border-[var(--border)] bg-[var(--titlebar-bg)] px-3 z-50 [-webkit-app-region:drag]"
      onDoubleClick={handleHeaderDoubleClick}
    >
      {/* Left: Navigation actions */}
      <div className="flex items-center gap-1 shrink-0 [-webkit-app-region:no-drag]">
        <button
          type="button"
          className="titlebar-btn"
          onClick={goBack}
          title={t("Back")}
          aria-label={t("Back")}
        >
          <ChevronLeft className="w-4 h-4 text-[var(--muted)]" />
        </button>
        <button
          type="button"
          className="titlebar-btn"
          onClick={goForward}
          title={t("Forward")}
          aria-label={t("Forward")}
        >
          <ChevronRight className="w-4 h-4 text-[var(--muted)]" />
        </button>
        <button
          type="button"
          className="titlebar-btn"
          onClick={() => window.location.reload()}
          title={t("Reload")}
          aria-label={t("Reload")}
        >
          <RefreshCcw className="w-3.5 h-3.5 text-[var(--muted)]" />
        </button>
      </div>

      {/* Center: Draggable App Title */}
      <div className="flex min-w-0 flex-1 items-center justify-center overflow-hidden pointer-events-none">
        <span className="flex min-w-0 items-center gap-2 text-xs font-medium text-[var(--muted)] tracking-wider">
          <span className="truncate text-[var(--text)]">Applio</span>
          <span className="hidden shrink-0 text-[10px] text-[var(--muted)] font-normal min-[420px]:inline">
            v3.6
          </span>
        </span>
      </div>

      {/* Right: Window Controls (minimize/close need the desktop shell) */}
      <div className="flex items-center gap-1 shrink-0 [-webkit-app-region:no-drag] justify-end">
        {hasBridge && (
          <button
            type="button"
            className="titlebar-btn"
            onClick={handleMinimize}
            title={t("Minimize")}
            aria-label={t("Minimize")}
          >
            <Minus className="w-3.5 h-3.5 text-[var(--muted)]" />
          </button>
        )}
        <button
          type="button"
          className="titlebar-btn"
          onClick={handleMaximize}
          title={hasBridge ? (maximized ? t("Restore") : t("Maximize")) : t("Toggle fullscreen")}
          aria-label={hasBridge ? (maximized ? t("Restore") : t("Maximize")) : t("Toggle fullscreen")}
        >
          {maximized ? (
            <Minimize2 className="w-3.5 h-3.5 text-[var(--muted)]" />
          ) : (
            <Maximize2 className="w-3.5 h-3.5 text-[var(--muted)]" />
          )}
        </button>
        {hasBridge && (
          <button
            type="button"
            className="titlebar-btn titlebar-btn-close"
            onClick={handleClose}
            title={t("Close")}
            aria-label={t("Close")}
          >
            <X className="w-3.5 h-3.5 text-[var(--muted)]" />
          </button>
        )}
      </div>
    </header>
  );
}
