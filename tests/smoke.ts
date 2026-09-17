// Applio smoke suite (node:test + fetch, no dependencies). Run: npm test

import assert from "node:assert/strict";
import { type ChildProcess, execFile, spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { after, before, test } from "node:test";
import { fileURLToPath } from "node:url";

const ROOT: string = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PORT = 8123;
const BASE = `http://127.0.0.1:${PORT}`;
let api: ChildProcess | null = null;

interface ApiResponse {
  status: number;
  json: Record<string, unknown>;
}

async function req(method: string, p: string, body?: unknown): Promise<ApiResponse> {
  const r = await fetch(`${BASE}${p}`, {
    method,
    headers: body !== undefined ? { "content-type": "application/json" } : {},
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const json = (await r.json().catch(() => ({}))) as Record<string, unknown>;
  return { status: r.status, json };
}

function findPython(): string {
  if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;
  const candidates =
    process.platform === "win32"
      ? [
          path.join(ROOT, ".venv", "Scripts", "python.exe"),
          path.join(ROOT, "venv", "Scripts", "python.exe"),
          path.join(ROOT, "env", "Scripts", "python.exe"),
        ]
      : [
          path.join(ROOT, ".venv", "bin", "python"),
          path.join(ROOT, "venv", "bin", "python"),
          path.join(ROOT, "env", "bin", "python"),
        ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return process.platform === "win32" ? "python" : "python3";
}

before(async () => {
  const pythonBin = findPython();
  api = spawn(process.execPath, ["app/api/dist/index.js"], {
    cwd: ROOT,
    env: { ...process.env, API_PORT: String(PORT), APPLIO_ROOT: ROOT, PYTHON_BIN: pythonBin },
    windowsHide: true,
  });
  const deadline = Date.now() + 30000;
  for (;;) {
    try {
      const r = await fetch(`${BASE}/api/health`);
      if (r.ok) return;
    } catch {
      /* not up yet */
    }
    if (Date.now() > deadline) throw new Error("API did not start in 30s");
    await new Promise((r) => setTimeout(r, 500));
  }
});

after(() => {
  api?.kill();
});

test("health reports ok", async () => {
  const { status, json } = await req("GET", "/api/health");
  assert.equal(status, 200);
  assert.equal(json.ok, true);
});

test("setup status reports checks", async () => {
  const { status, json } = await req("GET", "/api/setup/status");
  assert.equal(status, 200);
  assert.ok(Array.isArray(json.checks));
  assert.equal(typeof json.ready, "boolean");
});

test("models endpoint returns lists", async () => {
  const { status, json } = await req("GET", "/api/models");
  assert.equal(status, 200);
  assert.ok(Array.isArray(json.models) && Array.isArray(json.indexes) && Array.isArray(json.audios));
});

test("models library and inspection endpoints", async () => {
  const { status, json } = await req("GET", "/api/models/library");
  assert.equal(status, 200);
  assert.ok(Array.isArray(json.models));

  const del = await req("DELETE", "/api/models/nonexistent-model-xyz");
  assert.equal(del.status, 404);

  const insp = await req("POST", "/api/models/inspect", {});
  assert.equal(insp.status, 400);
});

test("train pipeline endpoint validation", async () => {
  const r = await req("POST", "/api/train/pipeline", { modelName: "" });
  assert.equal(r.status, 400);
});

test("presets CRUD round-trip", async () => {
  const values = { pitch: 0, index_rate: 0.5, rms_mix_rate: 1, protect: 0.25 };
  let r = await req("POST", "/api/presets", { name: "__smoke__", values });
  assert.equal(r.status, 200);
  r = await req("GET", "/api/presets");
  const presets = r.json.presets as Array<{ name: string }>;
  assert.ok(presets.some((p) => p.name === "__smoke__"));
  r = await req("DELETE", "/api/presets/__smoke__");
  assert.equal(r.status, 200);
  r = await req("GET", "/api/presets");
  assert.ok(!(r.json.presets as Array<{ name: string }>).some((p) => p.name === "__smoke__"));
});

test("tts voices catalog loads", async () => {
  const { status, json } = await req("GET", "/api/tts/voices");
  assert.equal(status, 200);
  assert.ok((json.voices as unknown[]).length > 100);
});

test("train discovery endpoints", async () => {
  for (const p of [
    "/api/train/datasets",
    "/api/train/pretraineds",
    "/api/train/exports",
    "/api/train/gpus",
  ]) {
    const { status } = await req("GET", p);
    assert.equal(status, 200, p);
  }
});

test("settings get/put round-trip (restores value)", async () => {
  const beforeCfg = await req("GET", "/api/settings");
  assert.equal(beforeCfg.status, 200);
  const config = beforeCfg.json.config as Record<string, unknown>;
  const current = (config.model_index_filter as boolean) ?? false;
  const put = await req("PUT", "/api/settings", { model_index_filter: current });
  assert.equal(put.status, 200);
  assert.equal((put.json.config as Record<string, unknown>).model_index_filter, current);
});

test("misc status endpoints", async () => {
  for (const p of [
    "/api/tensorboard/status",
    "/api/report/info",
    "/api/plugins",
    "/api/realtime/status",
    "/api/realtime/config",
    "/api/jobs",
  ]) {
    const { status } = await req("GET", p);
    assert.equal(status, 200, p);
  }
});

test("validation errors are 400s, unknown jobs 404", async () => {
  assert.equal((await req("POST", "/api/download", { modelLink: "not-a-url" })).status, 400);
  assert.equal((await req("POST", "/api/extra/model-info", { pthPath: "nope/none.pth" })).status, 400);
  assert.equal((await req("POST", "/api/train/preprocess", { modelName: "" })).status, 400);
  assert.equal((await req("POST", "/api/jobs/does-not-exist/stop")).status, 404);
  assert.equal((await req("PUT", "/api/settings", { precision: "fp8" })).status, 400);
});

test("no gradio imports remain in python engine", async () => {
  const hits: string[] = [];
  const walk = (dir: string): void => {
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      if ([".venv", "env", "node_modules", ".git", "__pycache__"].includes(e.name)) continue;
      const full = path.join(dir, e.name);
      if (e.isDirectory()) walk(full);
      else if (e.name.endsWith(".py")) {
        const src = fs.readFileSync(full, "utf-8");
        if (/^\s*(import gradio|from gradio)/m.test(src)) hits.push(path.relative(ROOT, full));
      }
    }
  };
  walk(ROOT);
  assert.deepEqual(hits, []);
});

test("requirements + notebooks have no gradio runtime", async () => {
  const reqTxt = fs.readFileSync(path.join(ROOT, "requirements.txt"), "utf-8");
  assert.ok(!/^gradio==/m.test(reqTxt), "gradio still in requirements.txt");
  for (const nb of ["Applio.ipynb", "Applio_Kaggle.ipynb", "Applio_NoUI.ipynb"]) {
    const raw = fs.readFileSync(path.join(ROOT, "assets", nb), "utf-8");
    JSON.parse(raw); // still valid JSON
    assert.ok(!raw.includes("app.py"), `${nb} still launches app.py`);
  }
});

test("core.py exposes the f0-curve command", async () => {
  const python = findPython();
  const out = await new Promise<string>((resolve, reject) => {
    execFile(python, ["core.py", "--help"], { cwd: ROOT }, (err, stdout, stderr) =>
      err ? reject(new Error(stderr || String(err))) : resolve(stdout),
    );
  });
  assert.ok(out.includes("f0-curve"));
});
