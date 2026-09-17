// Post-build: copy client static assets + public dir next to the standalone
// server (Next.js omits them from standalone output). Run from app/web.
import fs from "node:fs";
import path from "node:path";

const webDir: string = path.resolve(import.meta.dirname, "..");
const repoRoot: string = path.resolve(webDir, "..", "..");
const standaloneDir: string = path.join(webDir, ".next", "standalone");

function copyDir(src: string, dest: string): void {
  if (!fs.existsSync(src)) return;
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const from = path.join(src, entry.name);
    const to = path.join(dest, entry.name);
    if (entry.isDirectory()) copyDir(from, to);
    else fs.copyFileSync(from, to);
  }
}

if (!fs.existsSync(standaloneDir)) {
  console.log("[copy-static] no standalone output, skipping");
  process.exit(0);
}

// 1. Copy static assets and public directory
copyDir(path.join(webDir, ".next", "static"), path.join(standaloneDir, ".next", "static"));
copyDir(path.join(webDir, "public"), path.join(standaloneDir, "public"));

// 2. Ensure hoisted monorepo dependencies (e.g. react) are present in standalone/node_modules
const standaloneModules = path.join(standaloneDir, "node_modules");
const rootModules = path.join(repoRoot, "node_modules");
const essentialDeps = ["react", "react-is"];

for (const dep of essentialDeps) {
  const target = path.join(standaloneModules, dep);
  const src = path.join(rootModules, dep);
  if (!fs.existsSync(target) && fs.existsSync(src)) {
    console.log(`[copy-static] copying hoisted ${dep} to standalone/node_modules/${dep}`);
    copyDir(src, target);
  }
}

console.log("[copy-static] static assets and standalone dependencies ready");
