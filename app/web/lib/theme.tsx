"use client";

import { type ReactNode, useEffect } from "react";
import { apiGet } from "./api";

export interface ThemeFile {
  name?: string;
  version?: string;
  description?: string;
  mode?: string;
  colors?: Record<string, string>;
  fonts?: { display?: string[]; body?: string[]; mono?: string[] };
  radius?: Record<string, string>;
  shadows?: Record<string, string>;
  titlebar?: { bg?: string; closeBg?: string; closeText?: string };
  custom_css?: string;
}

// Theme JSON keys -> CSS variables in globals.css
const COLOR_TO_VAR: Record<string, string> = {
  primary: "--accent",
  primarySoft: "--accent-soft",
  background: "--bg",
  surface: "--surface",
  surfaceDim: "--input-bg",
  panel: "--panel",
  border: "--border",
  text: "--text",
  heading: "--heading",
  muted: "--muted",
  ctaBg: "--cta-bg",
  ctaText: "--cta-text",
  buttonBg: "--button-bg",
  buttonBgHover: "--button-bg-hover",
  buttonTextHover: "--button-text-hover",
  buttonGhostText: "--button-ghost-text",
  buttonGhostBorder: "--button-ghost-border",
  buttonGhostBorderHover: "--button-ghost-border-hover",
  sliderTrack: "--slider-track",
  sliderThumb: "--slider-thumb",
  checkboxBorder: "--checkbox-border",
  checkboxBorderHover: "--checkbox-border-hover",
  checkboxChecked: "--checkbox-checked",
  checkboxCheck: "--checkbox-check",
  fileButtonBg: "--file-button-bg",
  fileButtonBorder: "--file-button-border",
  fileButtonText: "--file-button-text",
  fileButtonBgHover: "--file-button-bg-hover",
  focusBorder: "--focus-border",
  selectionBg: "--selection-bg",
  selectionText: "--selection-text",
  logBg: "--log-bg",
  ok: "--ok",
  warn: "--warn",
  err: "--err",
  dangerBg: "--danger-bg",
  dangerBorder: "--danger-border",
  dangerText: "--danger-text",
  dangerBgHover: "--danger-bg-hover",
  dangerBorderHover: "--danger-border-hover",
  dangerTextHover: "--danger-text-hover",
};

const RADIUS_TO_VAR: Record<string, string> = {
  card: "--radius-card",
  input: "--radius-input",
  button: "--radius-button",
  pill: "--radius-pill",
};

const SHADOW_TO_VAR: Record<string, string> = {
  card: "--shadow-card",
  button: "--shadow-button",
};

// Families prefixed with "google:" are loaded from Google Fonts on demand
// (Gradio GoogleFont parity). Everything else is used as a system stack.
function fontStack(families: string[] | undefined): { css: string; google: string[] } {
  if (!families || families.length === 0) return { css: "", google: [] };
  const google: string[] = [];
  const stack = families
    .map((f) => {
      if (f.startsWith("google:")) {
        const name = f.slice("google:".length);
        google.push(name);
        return `"${name}"`;
      }
      return f.includes(" ") && !f.startsWith('"') ? `"${f}"` : f;
    })
    .join(", ");
  return { css: stack, google };
}

function ensureGoogleFont(name: string): void {
  const id = `applio-font-${name.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
  if (document.getElementById(id)) return;
  const link = document.createElement("link");
  link.id = id;
  link.rel = "stylesheet";
  link.href = `https://fonts.googleapis.com/css2?family=${encodeURIComponent(name)}:wght@400;600;700&display=swap`;
  document.head.appendChild(link);
}

export function applyTheme(theme: ThemeFile): void {
  const root = document.documentElement;
  for (const [key, value] of Object.entries(theme.colors || {})) {
    const v = COLOR_TO_VAR[key];
    if (v && typeof value === "string") root.style.setProperty(v, value);
  }
  if (theme.mode) {
    root.dataset.themeMode = theme.mode;
    root.style.colorScheme = theme.mode === "light" ? "light" : "dark";
  }
  const fonts = theme.fonts || {};
  const display = fontStack(fonts.display);
  const body = fontStack(fonts.body);
  const mono = fontStack(fonts.mono);
  for (const f of [...display.google, ...body.google, ...mono.google]) ensureGoogleFont(f);
  if (display.css) root.style.setProperty("--font-display", display.css);
  if (body.css) {
    root.style.setProperty("--font-sans", body.css);
    document.body.style.fontFamily = body.css;
  }
  if (mono.css) root.style.setProperty("--font-mono", mono.css);
  for (const [key, value] of Object.entries(theme.radius || {})) {
    const v = RADIUS_TO_VAR[key];
    if (v && typeof value === "string") root.style.setProperty(v, value);
  }
  for (const [key, value] of Object.entries(theme.shadows || {})) {
    const v = SHADOW_TO_VAR[key];
    if (v && typeof value === "string") root.style.setProperty(v, value);
  }
  const titlebar = theme.titlebar || {};
  if (typeof titlebar.bg === "string") root.style.setProperty("--titlebar-bg", titlebar.bg);
  if (typeof titlebar.closeBg === "string") root.style.setProperty("--titlebar-close-bg", titlebar.closeBg);
  if (typeof titlebar.closeText === "string")
    root.style.setProperty("--titlebar-close-text", titlebar.closeText);
  // Gradio custom_css parity: raw CSS appended last, wins over tokens.
  let tag = document.getElementById("applio-theme-css");
  if (theme.custom_css) {
    if (!tag) {
      tag = document.createElement("style");
      tag.id = "applio-theme-css";
      document.head.appendChild(tag);
    }
    tag.textContent = theme.custom_css;
  } else {
    tag?.remove();
  }
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  useEffect(() => {
    let live = true;
    const load = () => {
      apiGet<{ id: string; theme: ThemeFile }>("/api/settings/theme", { force: true })
        .then((r) => {
          if (live && r.theme) applyTheme(r.theme);
        })
        .catch(() => {});
    };
    load();
    // Re-apply when Settings saves a new theme (no reload needed).
    window.addEventListener("applio:theme-changed", load);
    return () => {
      live = false;
      window.removeEventListener("applio:theme-changed", load);
    };
  }, []);
  return <>{children}</>;
}
