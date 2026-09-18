const fs = require("node:fs");
const path = require("node:path");

// Recursive copy that dereferences symlinks (cp -rL equivalent).
// Required because Next standalone tracing under pnpm leaves symlinks like
// standalone/node_modules/react -> ../../../node_modules/.pnpm/... which are
// broken inside the packaged app. Broken links are skipped with a warning.
function copyDereferenced(src, dest) {
  const st = fs.lstatSync(src, { throwIfNoEntry: false });
  if (!st) return;
  if (st.isSymbolicLink()) {
    let real;
    try {
      real = fs.realpathSync(src);
    } catch {
      console.warn(`[afterPack] skipping broken symlink: ${src}`);
      return;
    }
    return copyDereferenced(real, dest);
  }
  if (st.isDirectory()) {
    fs.mkdirSync(dest, { recursive: true });
    for (const entry of fs.readdirSync(src)) {
      copyDereferenced(path.join(src, entry), path.join(dest, entry));
    }
    return;
  }
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.copyFileSync(src, dest);
}

function findPackagedAppDir(appOutDir, productFilename) {
  const candidates = [
    // Linux / Windows (also covers asar:false layout)
    path.join(appOutDir, "resources", "app"),
    // macOS
    path.join(appOutDir, `${productFilename}.app`, "Contents", "Resources", "app"),
  ];
  for (const dir of candidates) {
    if (fs.existsSync(path.join(dir, "app", "web", ".next", "standalone", "server.js"))) return dir;
  }
  return null;
}

function ensureStandaloneNodeModules(packagedAppDir) {
  // electron-builder only ships production dependencies of @applio/desktop
  // (docs: https://www.electron.build/docs/contents — "only production
  // dependencies are included"). The Next standalone runtime (next, react,
  // styled-jsx, @swc/helpers, ...) belongs to @applio/web, so its
  // standalone/node_modules tree is pruned from the package even when listed
  // in `files`. Copy the already-dereferenced tree (prepared by
  // app/web/scripts/copy-static.mts, npm/pnpm compatible) here instead.
  // afterPack runs after files are packaged, before signing, which is the
  // documented stage for modifying the bundle structure.
  const src = path.resolve(__dirname, "..", "web", ".next", "standalone", "node_modules");
  const dest = path.join(packagedAppDir, "app", "web", ".next", "standalone", "node_modules");
  if (!fs.existsSync(src)) {
    console.warn("[afterPack] standalone node_modules source missing, skipping:", src);
    return;
  }
  if (fs.existsSync(path.join(dest, "next", "package.json"))) {
    console.log("[afterPack] standalone node_modules already present, skipping copy.");
    return;
  }
  console.log(`[afterPack] copying standalone node_modules to packaged app…`);
  copyDereferenced(src, dest);
  const ok = fs.existsSync(path.join(dest, "next", "package.json"));
  console.log(`[afterPack] standalone node_modules ${ok ? "ready" : "STILL MISSING — check build!"}`);
}

exports.default = async (context) => {
  const appOutDir = context.appOutDir;
  const productFilename = context.packager.appInfo.productFilename;
  const packagedAppDir = findPackagedAppDir(appOutDir, productFilename);
  if (!packagedAppDir) {
    console.warn("[afterPack] packaged app dir not found under:", appOutDir);
  } else {
    ensureStandaloneNodeModules(packagedAppDir);
  }

  if (context.electronPlatformName !== "win32") return;
  const exeName = `${productFilename}.exe`;
  const exePath = path.join(appOutDir, exeName);
  const root = path.resolve(__dirname, "..", "..");
  const iconPath = path.join(root, "assets", "ICON.ico");
  const version = context.packager.appInfo.version || "3.6.4";

  if (!fs.existsSync(exePath)) {
    console.warn("[afterPack] Target exe not found:", exePath);
    return;
  }

  try {
    console.log(`[afterPack] Stamping ${exeName} with Applio icon and metadata (v${version})…`);
    const { rcedit } = require("rcedit");
    await rcedit(exePath, {
      icon: iconPath,
      "file-version": version,
      "product-version": version,
      "version-string": {
        CompanyName: "Applio",
        FileDescription: "Applio",
        ProductName: "Applio",
        LegalCopyright: "Copyright © Applio contributors",
      },
    });
    console.log(`[afterPack] Successfully stamped ${exeName} with Applio icon.`);
  } catch (err) {
    console.warn("[afterPack] Warning: rcedit stamping failed:", err.message);
  }
};
