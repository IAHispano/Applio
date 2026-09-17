"use client";

import { AlertCircle, CheckCircle2, X } from "lucide-react";
import { useEffect, useState } from "react";

// Minimal gr.Info/gr.Warning/gr.Error parity: transient stacked toasts.
// Success/info call sites dispatch applio:toast; form errors stay inline.
export type ToastKind = "info" | "error";

interface Toast {
  id: number;
  kind: ToastKind;
  text: string;
}

let nextId = 1;

export function toast(text: string, kind: ToastKind = "info"): void {
  window.dispatchEvent(new CustomEvent("applio:toast", { detail: { text, kind } }));
}

export default function Toaster() {
  const [items, setItems] = useState<Toast[]>([]);

  useEffect(() => {
    const onToast = (e: Event) => {
      const { text, kind } = (e as CustomEvent).detail as { text: string; kind: ToastKind };
      const id = nextId++;
      setItems((prev) => [...prev.slice(-3), { id, kind, text }]);
      setTimeout(() => setItems((prev) => prev.filter((t) => t.id !== id)), 4500);
    };
    window.addEventListener("applio:toast", onToast);
    return () => window.removeEventListener("applio:toast", onToast);
  }, []);

  const dismiss = (id: number) => {
    setItems((prev) => prev.filter((t) => t.id !== id));
  };

  if (items.length === 0) return null;

  return (
    <section
      aria-label="Notifications"
      className="fixed right-4 bottom-4 z-50 flex flex-col gap-2 pointer-events-none max-w-sm w-full"
    >
      {items.map((t) => {
        const isError = t.kind === "error";
        return (
          <div
            key={t.id}
            role={isError ? "alert" : "status"}
            aria-live={isError ? "assertive" : "polite"}
            className={`pointer-events-auto flex items-start gap-3 p-3.5 rounded-xl border backdrop-blur-md shadow-xl text-xs transition-all ${
              isError
                ? "bg-red-950/90 border-red-500/40 text-red-200"
                : "bg-neutral-900/95 border-white/20 text-neutral-100"
            }`}
          >
            <span className="shrink-0 mt-0.5" aria-hidden="true">
              {isError ? (
                <AlertCircle size={15} className="text-red-400" />
              ) : (
                <CheckCircle2 size={15} className="text-emerald-400" />
              )}
            </span>
            <span className="flex-1 font-medium leading-relaxed">{t.text}</span>
            <button
              type="button"
              onClick={() => dismiss(t.id)}
              aria-label="Dismiss notification"
              className="shrink-0 -mr-1 -mt-1 p-1 rounded-md text-neutral-400 hover:text-white hover:bg-white/10 transition-colors"
            >
              <X size={13} aria-hidden="true" />
            </button>
          </div>
        );
      })}
    </section>
  );
}
