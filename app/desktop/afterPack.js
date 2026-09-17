const fs = require("node:fs");
const path = require("node:path");

exports.default = async (context) => {
  if (context.electronPlatformName !== "win32") return;
  const exeName = `${context.packager.appInfo.productFilename}.exe`;
  const exePath = path.join(context.appOutDir, exeName);
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
