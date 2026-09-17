"use client";

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
  if (items.length === 0) return null;
  return (
    <div style={{ position: "fixed", right: 16, bottom: 16, zIndex: 60, display: "grid", gap: 8 }}>
      {items.map((t) => (
        <div
          key={t.id}
          className="card"
          style={{
            margin: 0,
            borderColor: t.kind === "error" ? "var(--err)" : undefined,
            maxWidth: 360,
          }}
          role="status"
        >
          {t.text}
        </div>
      ))}
    </div>
  );
}
