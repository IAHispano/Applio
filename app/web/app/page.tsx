"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import JobPanel from "../components/JobPanel";
import { MENU } from "../components/layout/nav";
import { apiGet, apiSend, errMsg } from "../lib/api";

interface SetupCheck {
  id: string;
  label: string;
  status: "ok" | "missing" | "warn";
  detail: string;
}

interface SetupStatus {
  ready: boolean;
  checks: SetupCheck[];
  checkedAt: string;
}

function StatusDot({ status }: { status: SetupCheck["status"] }) {
  const color = status === "ok" ? "var(--ok)" : status === "warn" ? "var(--warn)" : "var(--err)";
  return <span style={{ width: 8, height: 8, borderRadius: 999, background: color, flexShrink: 0 }} />;
}

export default function Home() {
  const router = useRouter();
  const [status, setStatus] = useState<SetupStatus | null>(null);
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async (force = false) => {
    try {
      setStatus(await apiGet<SetupStatus>(`/api/setup/status${force ? "?refresh=1" : ""}`));
      setError("");
    } catch (e) {
      setError(errMsg(e));
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function install() {
    setError("");
    setBusy(true);
    try {
      const { jobId: id } = await apiSend<{ jobId: string }>("/api/setup/install", "POST");
      setJobId(id);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  }

  async function prerequisites() {
    setError("");
    try {
      const { jobId: id } = await apiSend<{ jobId: string }>("/api/setup/prerequisites", "POST");
      setJobId(id);
    } catch (e) {
      setError(errMsg(e));
    }
  }

  const ready = status?.ready ?? false;

  return (
    <div className="h-full flex flex-col gap-4 overflow-auto pb-4">
      <div className="flex items-end justify-between gap-4 flex-wrap">
        <div>
          <h1 className="title font-bold leading-none" style={{ fontSize: 44, margin: 0 }}>
            Applio
          </h1>
          <p className="muted" style={{ margin: "6px 0 0" }}>
            High-quality voice conversion, on your machine.
          </p>
        </div>
        <div className="row">
          <span className={`badge ${ready ? "done" : "error"}`}>{ready ? "Ready" : "Setup needed"}</span>
          {ready ? (
            <button type="button" className="cta" onClick={() => router.push("/inference")}>
              Start converting →
            </button>
          ) : (
            <button type="button" className="cta" onClick={install} disabled={busy}>
              {busy ? "Starting…" : "Install / Repair"}
            </button>
          )}
        </div>
      </div>

      <div className="grid gap-4" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))" }}>
        {MENU.filter((m) => m.to !== "/").map((m) => {
          const Icon = m.icon;
          return (
            <Link
              key={m.to}
              href={m.to}
              className="border border-white/10 rounded-xl p-4 hover:bg-white/10 slow transition-colors duration-200"
            >
              <Icon className="w-6 h-6 opacity-70" />
              <h3 className="text-neutral-200 font-semibold title mt-2" style={{ marginBottom: 2 }}>
                {m.label}
              </h3>
              <p className="text-neutral-400 text-xs" style={{ margin: 0 }}>
                {m.blurb}
              </p>
            </Link>
          );
        })}
      </div>

      <div className="card !mb-0">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2 className="title" style={{ margin: 0 }}>
            System status
          </h2>
          <div className="row">
            <button type="button" className="ghost" onClick={() => refresh(true)}>
              Re-check
            </button>
            <button type="button" className="ghost" onClick={prerequisites}>
              Engine models
            </button>
          </div>
        </div>
        {error && <p style={{ color: "var(--err)" }}>{error}</p>}
        {!status && !error && <p className="muted">Contacting the API…</p>}
        {status && (
          <div className="grid2" style={{ marginTop: 12 }}>
            {status.checks.map((c) => (
              <div className="row" key={c.id}>
                <StatusDot status={c.status} />
                <strong style={{ fontSize: 13 }}>{c.label}</strong>
                <span className="muted">{c.detail}</span>
              </div>
            ))}
          </div>
        )}
      </div>
      <JobPanel jobId={jobId} />
    </div>
  );
}
