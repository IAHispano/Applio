"use client";

import { createContext, type ReactNode, useContext, useEffect, useState } from "react";
import { apiGet } from "./api";

// Minimal Gradio I18nAuto parity: the API resolves the active language
// (settings override, else OS locale) and serves its dictionary from
// assets/i18n/languages/*.json. t(key) falls back to the key itself,
// which is English by convention — exactly like i18n("...") in app.py.
type TFn = (key: string) => string;

const I18nCtx = createContext<{ t: TFn; code: string }>({ t: (k) => k, code: "en_US" });

export function useI18n(): { t: TFn; code: string } {
  return useContext(I18nCtx);
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [dict, setDict] = useState<Record<string, string>>({});
  const [code, setCode] = useState("en_US");
  useEffect(() => {
    let live = true;
    const load = () => {
      apiGet<{ code: string; dict: Record<string, string> }>("/api/settings/language", { force: true })
        .then((r) => {
          if (!live) return;
          setCode(r.code || "en_US");
          setDict(r.dict || {});
        })
        .catch(() => {});
    };
    load();
    // Refetch when Settings saves a new language (no full reload needed).
    window.addEventListener("applio:language-changed", load);
    return () => {
      live = false;
      window.removeEventListener("applio:language-changed", load);
    };
  }, []);
  const t: TFn = (key) => dict[key] ?? key;
  return <I18nCtx.Provider value={{ t, code }}>{children}</I18nCtx.Provider>;
}
