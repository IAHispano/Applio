// Post-build: copy client static assets + public dir next to the standalone
// server (Next.js omits them from standalone output). Run from app/web.
// Compatible with both npm (flat node_modules, real dirs) and pnpm
// (symlinked node_modules, standalone tracing can leave broken symlinks).
// Docs (verified with Playwright, HTTP 200):
// - Next.js standalone: https://nextjs.org/docs/app/api-reference/config/next-config-js/output
// - pnpm symlinks: https://pnpm.io/motivation
// - electron-builder FileSet: https://www.electron.build/docs/api/app-builder-lib.interface.fileset/
import fs from "node:fs";
import path from "node:path";

const webDir: string = path.resolve(import.meta.dirname, "..");
const repoRoot: string = path.resolve(webDir, "..", "..");
const standaloneDir: string = path.join(webDir, ".next", "standalone");

// Detect package-manager layout for logging (best-effort, never fails build).
function detectLayout(rootModules: string): "pnpm" | "npm" | "unknown" {
  // Mixed installs happen (npm + pnpm in same repo): .pnpm store wins.
  if (fs.existsSync(path.join(rootModules, ".pnpm"))) return "pnpm";
  try {
    const probe = path.join(rootModules, "react");
    const st = fs.lstatSync(probe, { throwIfNoEntry: false });
    if (st?.isSymbolicLink()) return "pnpm";
    if (st?.isDirectory()) return "npm";
  } catch {
    // ignore
  }
  return "unknown";
}

function ensureDir(dest: string): void {
  const st = fs.lstatSync(dest, { throwIfNoEntry: false });
  // A (possibly broken) symlink blocks mkdirSync with ENOENT/EEXIST.
  if (st?.isSymbolicLink()) fs.unlinkSync(dest);
  fs.mkdirSync(dest, { recursive: true });
}

function copyDir(src: string, dest: string): void {
  // Dereference src in case the caller passes a symlinked path (pnpm).
  let realSrc = src;
  const srcStat = fs.lstatSync(src, { throwIfNoEntry: false });
  if (!srcStat) return;
  if (srcStat.isSymbolicLink()) {
    try {
      realSrc = fs.realpathSync(src);
    } catch {
      console.warn(`[copy-static] skipping broken symlink src: ${src}`);
      return;
    }
  }
  if (!fs.existsSync(realSrc)) return;
  ensureDir(dest);
  for (const entry of fs.readdirSync(realSrc, { withFileTypes: true })) {
    const from = path.join(realSrc, entry.name);
    const to = path.join(dest, entry.name);
    if (entry.isSymbolicLink()) {
      // Dereference (cp -rL equivalent): copy real file/dir, never the link.
      // Required for Electron AppImage: links pointing to ../../.pnpm break.
      let realFrom: string;
      try {
        realFrom = fs.realpathSync(from);
      } catch {
        console.warn(`[copy-static] skipping broken symlink: ${from}`);
        continue;
      }
      const realStat = fs.statSync(realFrom);
      if (realStat.isDirectory()) copyDir(realFrom, to);
      else {
        ensureDir(path.dirname(to));
        fs.copyFileSync(realFrom, to);
      }
    } else if (entry.isDirectory()) {
      copyDir(from, to);
    } else {
      fs.copyFileSync(from, to);
    }
  }
}

if (!fs.existsSync(standaloneDir)) {
  console.log("[copy-static] no standalone output, skipping");
  process.exit(0);
}

// 1. Copy static assets and public directory
copyDir(path.join(webDir, ".next", "static"), path.join(standaloneDir, ".next", "static"));
copyDir(path.join(webDir, "public"), path.join(standaloneDir, "public"));

// 2. Ensure hoisted monorepo dependencies are present as REAL dirs in standalone.
const standaloneModules = path.join(standaloneDir, "node_modules");
const rootModules = path.join(repoRoot, "node_modules");
const webModules = path.join(webDir, "node_modules");
console.log(`[copy-static] layout: ${detectLayout(rootModules)}`);
ensureDir(standaloneModules);

const essentialDeps = ["react", "react-dom", "react-is", "next"];

// Next's own runtime deps (require-hook.js resolves styled-jsx at load,
// constants.js needs @swc/helpers). Under pnpm these live isolated in
// app/web/node_modules, not nested inside next/, so copy them explicitly.
// Read live from next/package.json so upgrades stay covered.
function nextRuntimeDeps(): string[] {
  for (const base of [webModules, rootModules]) {
    try {
      const pkgPath = path.join(base, "next", "package.json");
      if (!fs.existsSync(pkgPath)) continue;
      const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf8")) as { dependencies?: Record<string, string> };
      return Object.keys(pkg.dependencies ?? {});
    } catch {
      // ignore and try next base
    }
  }
  return ["@next/env", "@swc/helpers", "styled-jsx"];
}

function findDepSrc(dep: string): string | null {
  // npm: hoisted to repo root. pnpm: kept in app/web/node_modules.
  // Mixed installs: either can hold the real dir.
  for (const base of [webModules, rootModules]) {
    const candidate = path.join(base, dep);
    if (fs.existsSync(candidate)) return candidate;
  }
  return null;
}

function ensureRealDep(dep: string): void {
  const target = path.join(standaloneModules, dep);
  const src = findDepSrc(dep);
  if (!src) return; // not installed under this manager, skip
  const targetStat = fs.lstatSync(target, { throwIfNoEntry: false });
  if (targetStat?.isSymbolicLink()) {
    // Broken or valid link -> replace with real copy for Electron/Docker.
    const resolves = fs.existsSync(target); // follows link
    console.log(`[copy-static] replacing ${resolves ? "symlink" : "broken symlink"} ${dep}`);
    fs.unlinkSync(target);
  }
  if (!fs.existsSync(target)) {
    console.log(`[copy-static] copying hoisted ${dep} to standalone/node_modules/${dep}`);
    copyDir(src, target);
  }
}

// 2a. Repair any broken symlink Next tracing left behind (pnpm layout).
for (const entry of fs.readdirSync(standaloneModules, { withFileTypes: true })) {
  if (!entry.isSymbolicLink()) continue;
  const full = path.join(standaloneModules, entry.name);
  if (!fs.existsSync(full)) {
    console.log(`[copy-static] found broken symlink: ${entry.name}`);
    ensureRealDep(entry.name);
  }
}

// 2b. Ensure essentials + Next runtime closure as real dirs (npm flat + pnpm).
for (const dep of new Set([...essentialDeps, ...nextRuntimeDeps()])) ensureRealDep(dep);

console.log("[copy-static] static assets and standalone dependencies ready");
