"use client";

import {
  ArrowRight,
  Database,
  Download,
  FileCheck,
  FileX,
  Folder,
  Info,
  Layers,
  RefreshCw,
  Search,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import ModelInfoCard, { type ModelMetadata } from "../../components/models/ModelInfoCard";
import PageHeader from "../../components/layout/PageHeader";
import BlenderPanel from "../../components/models/BlenderPanel";
import DownloadPanel from "../../components/models/DownloadPanel";
import Modal from "../../components/ui/Modal";
import SegmentedControl from "../../components/ui/SegmentedControl";
import { apiGet, apiSend, errMsg } from "../../lib/api";
import { useI18n } from "../../lib/i18n";

interface ModelItem {
  id: string;
  name: string;
  pthPath: string;
  pthSize: number;
  indexPath: string | null;
  indexSize: number | null;
  modifiedAt: string;
  folder: string;
}

type Section = "library" | "download" | "blend" | "inspect";

function formatBytes(bytes: number): string {
  if (!bytes || bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / k ** i).toFixed(1)} ${sizes[i]}`;
}

export default function ModelsPage() {
  const router = useRouter();
  const { t } = useI18n();
  const [section, setSection] = useState<Section>("library");

  // Library state
  const [models, setModels] = useState<ModelItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<ModelItem | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Inspect modal state
  const [inspectModal, setInspectModal] = useState<ModelItem | null>(null);
  const [inspectMeta, setInspectMeta] = useState<ModelMetadata | null>(null);
  const [inspectLoading, setInspectLoading] = useState(false);
  const [inspectError, setInspectError] = useState("");

  // Custom path inspect (subtab)
  const [customPth, setCustomPth] = useState("");
  const [customMeta, setCustomMeta] = useState<ModelMetadata | null>(null);
  const [customLoading, setCustomLoading] = useState(false);
  const [customError, setCustomError] = useState("");

  const loadLibrary = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await apiGet<{ models: ModelItem[] }>("/api/models/library");
      setModels(res.models || []);
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (section === "library") {
      loadLibrary();
    }
  }, [section, loadLibrary]);

  async function openInspect(item: ModelItem) {
    setInspectModal(item);
    setInspectMeta(null);
    setInspectError("");
    setInspectLoading(true);
    try {
      const res = await apiSend<{ ok: boolean; metadata: ModelMetadata }>("/api/models/inspect", "POST", {
        pthPath: item.pthPath,
      });
      setInspectMeta(res.metadata);
    } catch (e) {
      setInspectError(errMsg(e));
    } finally {
      setInspectLoading(false);
    }
  }

  async function confirmDelete() {
    if (!deleteTarget) return;
    setDeleting(true);
    try {
      const nameToDelete = deleteTarget.folder !== "root" ? deleteTarget.folder : deleteTarget.name;
      await apiSend(`/api/models/${encodeURIComponent(nameToDelete)}`, "DELETE");
      setDeleteTarget(null);
      await loadLibrary();
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setDeleting(false);
    }
  }

  function openInInference(item: ModelItem) {
    const url = `/inference?model=${encodeURIComponent(item.pthPath)}${
      item.indexPath ? `&index=${encodeURIComponent(item.indexPath)}` : ""
    }`;
    router.push(url);
  }

  async function inspectCustom() {
    if (!customPth.trim()) {
      setCustomError(t("Please enter or select a .pth model path."));
      return;
    }
    setCustomError("");
    setCustomLoading(true);
    setCustomMeta(null);
    try {
      const res = await apiSend<{ ok: boolean; metadata: ModelMetadata }>("/api/models/inspect", "POST", {
        pthPath: customPth.trim(),
      });
      setCustomMeta(res.metadata);
    } catch (e) {
      setCustomError(errMsg(e));
    } finally {
      setCustomLoading(false);
    }
  }

  const filteredModels = models.filter(
    (m) =>
      m.name.toLowerCase().includes(search.toLowerCase()) ||
      m.folder.toLowerCase().includes(search.toLowerCase()) ||
      m.pthPath.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <PageHeader
        title={t("Voice Models")}
        description={t(
          "Manage your voice model collection, inspect checkpoint metadata, and blend or download weights.",
        )}
      >
        <SegmentedControl
          value={section}
          onChange={setSection}
          ariaLabel={t("Model sections")}
          tabPanels
          options={[
            { value: "library", label: t("Model Library"), icon: Database },
            { value: "download", label: t("Download Models"), icon: Download },
            { value: "blend", label: t("Voice Blender"), icon: Layers },
            { value: "inspect", label: t("Inspect Path"), icon: Info },
          ]}
        />
      </PageHeader>

      {error && (
        <div
          role="alert"
          aria-live="assertive"
          className="mb-4 p-3 rounded-lg border border-[var(--err)] text-[var(--err)] bg-[color-mix(in_srgb,var(--err)_10%,transparent)]"
        >
          {error}
        </div>
      )}

      {/* 1. MODEL LIBRARY VIEW */}
      {section === "library" && (
        <div id="panel-library" role="tabpanel" aria-labelledby="tab-library" className="space-y-4">
          {/* Controls bar: Search, Refresh, Download CTA */}
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <div className="flex items-center gap-2 flex-1 min-w-[240px] max-w-md bg-white/5 border border-white/10 rounded-lg px-3 py-2">
              <Search size={16} className="text-neutral-400" aria-hidden="true" />
              <input
                type="text"
                placeholder={t("Search models by name or folder…")}
                aria-label={t("Search models by name or folder")}
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="bg-transparent border-0 p-0 text-sm text-white focus:outline-none w-full"
              />
              {search && (
                <button
                  type="button"
                  onClick={() => setSearch("")}
                  aria-label={t("Clear search")}
                  className="text-neutral-400 hover:text-white"
                >
                  <X size={14} aria-hidden="true" />
                </button>
              )}
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                className="ghost flex items-center gap-1.5"
                onClick={loadLibrary}
                disabled={loading}
              >
                <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
                <span>{t("Refresh models and indexes")}</span>
              </button>

              <button
                type="button"
                className="cta flex items-center gap-1.5"
                onClick={() => setSection("download")}
              >
                <Download size={14} />
                <span>{t("Get Models")}</span>
              </button>
            </div>
          </div>

          {/* Model Cards Grid */}
          {filteredModels.length === 0 ? (
            <div className="card text-center py-12 space-y-4">
              <Database size={40} className="mx-auto text-neutral-500" />
              <div>
                <h3 className="text-lg font-semibold text-white m-0">{t("No voice models found")}</h3>
                <p className="text-sm text-neutral-400 m-0 max-w-md mx-auto mt-1">
                  {search
                    ? t("No models match your search.")
                    : t(
                        "Your models directory (logs/) is currently empty. Download community weights or train your own voice model to get started.",
                      )}
                </p>
              </div>
              <div className="flex justify-center gap-3 pt-2">
                <button
                  type="button"
                  className="cta flex items-center gap-2"
                  onClick={() => setSection("download")}
                >
                  <Download size={16} />
                  <span>{t("Download a Model")}</span>
                </button>
                <Link href="/train" className="inline-flex">
                  <button type="button" className="ghost">
                    {t("Train New Model")}
                  </button>
                </Link>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {filteredModels.map((m) => (
                <div
                  key={m.id}
                  className="card m-0 hover:border-white/20 transition-all flex flex-col justify-between"
                >
                  <div>
                    {/* Header: Title & Folder */}
                    <div className="flex items-start justify-between gap-2 mb-2">
                      <h3 className="font-semibold text-white text-base truncate m-0 flex-1" title={m.name}>
                        {m.name}
                      </h3>
                      <span className="text-xs px-2 py-0.5 rounded bg-white/10 text-neutral-300 flex items-center gap-1 shrink-0">
                        <Folder size={12} />
                        <span>{m.folder}</span>
                      </span>
                    </div>

                    {/* Stats & Index status */}
                    <div className="space-y-1.5 text-xs text-neutral-400 my-3">
                      <div className="flex justify-between">
                        <span>{t("Weights (.pth):")}</span>
                        <span className="text-neutral-200 tabular-nums">{formatBytes(m.pthSize)}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span>{t("Feature Index:")}</span>
                        {m.indexPath ? (
                          <span className="text-neutral-200 flex items-center gap-1">
                            <FileCheck size={12} />
                            <span>{formatBytes(m.indexSize || 0)}</span>
                          </span>
                        ) : (
                          <span className="text-neutral-500 flex items-center gap-1">
                            <FileX size={12} />
                            <span>{t("None")}</span>
                          </span>
                        )}
                      </div>
                      <div className="flex justify-between text-neutral-500 pt-1 border-t border-white/5">
                        <span>{t("Modified:")}</span>
                        <span>{new Date(m.modifiedAt).toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>

                  {/* Actions Footer */}
                  <div className="flex items-center justify-between gap-2 pt-3 border-t border-white/10 mt-2">
                    <button
                      type="button"
                      onClick={() => openInInference(m)}
                      className="cta text-xs px-3 py-1.5 flex items-center gap-1.5"
                    >
                      <Sparkles size={13} />
                      <span>{t("Use")}</span>
                    </button>

                    <div className="flex items-center gap-1">
                      <button
                        type="button"
                        onClick={() => openInspect(m)}
                        className="ghost text-xs px-2.5 py-1.5 flex items-center gap-1"
                        title={t("View checkpoint metadata")}
                        aria-label={`${t("Inspect model metadata for")} ${m.name}`}
                      >
                        <Info size={13} aria-hidden="true" />
                        <span>{t("Inspect")}</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => setDeleteTarget(m)}
                        className="danger text-xs px-2.5 py-1.5"
                        title={t("Delete model files")}
                        aria-label={`${t("Delete model")} ${m.name}`}
                      >
                        <Trash2 size={13} aria-hidden="true" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 2. DOWNLOAD PANEL */}
      {section === "download" && (
        <div id="panel-download" role="tabpanel" aria-labelledby="tab-download">
          <DownloadPanel />
        </div>
      )}

      {/* 3. VOICE BLENDER PANEL */}
      {section === "blend" && (
        <div id="panel-blend" role="tabpanel" aria-labelledby="tab-blend">
          <BlenderPanel />
        </div>
      )}

      {/* 4. INSPECT CUSTOM PATH */}
      {section === "inspect" && (
        <div id="panel-inspect" role="tabpanel" aria-labelledby="tab-inspect" className="space-y-4">
          <div className="card space-y-4">
            <div className="border-b border-white/10 pb-3.5 space-y-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Info size={18} className="text-white" />
                  <h2 className="text-base font-bold text-white m-0">
                    {t("Inspect Model File")}
                  </h2>
                </div>
              </div>
              <p className="text-xs text-neutral-400 m-0 leading-relaxed">
                {t(
                  "Enter any repository-relative or absolute path to a .pth checkpoint to read its architecture and training parameters.",
                )}
              </p>
            </div>
            <div className="flex flex-col sm:flex-row gap-3 max-w-xl">
              <label htmlFor="custom-pth-input" className="sr-only">
                {t("Path to .pth checkpoint")}
              </label>
              <input
                id="custom-pth-input"
                type="text"
                value={customPth}
                onChange={(e) => setCustomPth(e.target.value)}
                placeholder="logs/my-model/my-model.pth"
                className="flex-1 h-10 px-3 text-sm rounded-xl bg-white/5 border border-white/10"
              />
              <button
                type="button"
                className="cta h-10 px-4 flex items-center justify-center gap-2 text-sm font-medium rounded-xl shrink-0"
                onClick={inspectCustom}
              >
                <Info size={16} className="shrink-0" />
                <span>{t("Inspect File")}</span>
              </button>
            </div>
          </div>
          <ModelInfoCard
            metadata={customMeta}
            loading={customLoading}
            error={customError}
            pthPath={customPth}
          />
        </div>
      )}

      {/* INSPECT METADATA MODAL */}
      <Modal
        isOpen={!!inspectModal}
        onClose={() => setInspectModal(null)}
        title={inspectModal?.name || t("Model Metadata")}
        icon={<Info size={18} className="text-white" />}
      >
        {inspectLoading && <p className="muted text-sm">{t("Reading model checkpoint…")}</p>}
        {inspectError && (
          <div
            role="alert"
            aria-live="assertive"
            className="p-3 rounded-lg border border-[var(--err)] text-[var(--err)] bg-[color-mix(in_srgb,var(--err)_10%,transparent)]"
          >
            {inspectError}
          </div>
        )}

        {inspectMeta && (
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="bg-black/30 p-2.5 rounded-lg border border-white/5">
              <span className="text-neutral-400 text-xs block">{t("Model Name")}</span>
              <span className="font-medium text-white">{inspectMeta.model_name || t("None")}</span>
            </div>
            <div className="bg-black/30 p-2.5 rounded-lg border border-white/5">
              <span className="text-neutral-400 text-xs block">{t("Author")}</span>
              <span className="font-medium text-white">{inspectMeta.author || t("Anonymous")}</span>
            </div>
            <div className="bg-black/30 p-2.5 rounded-lg border border-white/5">
              <span className="text-neutral-400 text-xs block">{t("Epochs")}</span>
              <span className="font-medium text-white">{inspectMeta.epochs || t("None")}</span>
            </div>
            <div className="bg-black/30 p-2.5 rounded-lg border border-white/5">
              <span className="text-neutral-400 text-xs block">{t("Training Steps")}</span>
              <span className="font-medium text-white">{inspectMeta.step || t("None")}</span>
            </div>
            <div className="bg-black/30 p-2.5 rounded-lg border border-white/5">
              <span className="text-neutral-400 text-xs block">{t("Sampling Rate")}</span>
              <span className="font-medium text-white">{inspectMeta.sr || t("None")}</span>
            </div>
            <div className="bg-black/30 p-2.5 rounded-lg border border-white/5">
              <span className="text-neutral-400 text-xs block">{t("Pitch Guidance (F0)")}</span>
              <span className="font-medium text-white">
                {inspectMeta.f0 === "1" ? t("Yes") : inspectMeta.f0 || t("None")}
              </span>
            </div>
            <div className="bg-black/30 p-2.5 rounded-lg border border-white/5">
              <span className="text-neutral-400 text-xs block">{t("Vocoder")}</span>
              <span className="font-medium text-white">{inspectMeta.vocoder || "HiFi-GAN"}</span>
            </div>
            <div className="bg-black/30 p-2.5 rounded-lg border border-white/5">
              <span className="text-neutral-400 text-xs block">{t("Embedder Model")}</span>
              <span className="font-medium text-white">{inspectMeta.embedder_model || "contentvec"}</span>
            </div>
            <div className="col-span-2 bg-black/30 p-2.5 rounded-lg border border-white/5">
              <span className="text-neutral-400 text-xs block">{t("Creation Date")}</span>
              <span className="font-medium text-white">{inspectMeta.creation_date || t("Unknown")}</span>
            </div>
          </div>
        )}

        <div className="flex justify-end gap-2 pt-2 border-t border-white/10 mt-4">
          <button type="button" className="ghost" onClick={() => setInspectModal(null)}>
            {t("Close")}
          </button>
          <button
            type="button"
            className="cta flex items-center gap-1.5"
            onClick={() => {
              const m = inspectModal;
              setInspectModal(null);
              if (m) openInInference(m);
            }}
          >
            <span>{t("Use in Inference")}</span>
            <ArrowRight size={14} />
          </button>
        </div>
      </Modal>

      {/* DELETE CONFIRMATION MODAL */}
      <Modal
        isOpen={!!deleteTarget}
        onClose={() => setDeleteTarget(null)}
        title={t("Delete Model?")}
        size="sm"
        danger
      >
        <p className="text-sm text-neutral-300">
          {t("Are you sure you want to permanently delete")} <strong>{deleteTarget?.name}</strong>{" "}
          {t("from disk? This will remove its .pth and .index files.")}
        </p>
        <div className="flex justify-end gap-2 pt-4 border-t border-white/10 mt-4">
          <button type="button" className="ghost" onClick={() => setDeleteTarget(null)} disabled={deleting}>
            {t("Cancel")}
          </button>
          <button type="button" className="danger" onClick={confirmDelete} disabled={deleting}>
            {deleting ? t("Deleting…") : t("Delete Model")}
          </button>
        </div>
      </Modal>
    </div>
  );
}
