"use client";

import {
  ArrowRight,
  Calendar,
  Check,
  Cpu,
  Copy,
  FileCheck,
  Hash,
  Layers,
  Music,
  Sliders,
  Sparkles,
  User,
  X,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { fileBasename } from "../../lib/api";
import { useI18n } from "../../lib/i18n";
import { toast } from "../../lib/toast";

export interface ModelMetadata {
  model_name?: string;
  author?: string;
  epochs?: string;
  step?: string;
  sr?: string;
  f0?: string;
  vocoder?: string;
  embedder_model?: string;
  creation_date?: string;
  model_hash?: string;
  dataset_length?: string;
  speakers_id?: string | number;
}

interface ModelInfoCardProps {
  metadata: ModelMetadata | null;
  loading?: boolean;
  error?: string;
  pthPath?: string;
  onClose?: () => void;
  onUseInInference?: () => void;
}

export default function ModelInfoCard({
  metadata,
  loading = false,
  error = "",
  pthPath = "",
  onClose,
  onUseInInference,
}: ModelInfoCardProps) {
  const { t } = useI18n();
  const router = useRouter();
  const [copied, setCopied] = useState(false);

  if (!loading && !error && !metadata) return null;

  function copyHash(hashText: string) {
    if (!hashText) return;
    navigator.clipboard.writeText(hashText);
    setCopied(true);
    toast(t("Model hash copied to clipboard"));
    setTimeout(() => setCopied(false), 2000);
  }

  function handleUseInInference() {
    if (onUseInInference) {
      onUseInInference();
      return;
    }
    const path = pthPath || metadata?.model_name || "";
    if (path) {
      router.push(`/inference?model=${encodeURIComponent(path)}`);
    }
  }

  return (
    <div className="card space-y-5 animate-in fade-in duration-200" aria-label={t("Model Checkpoint Details")}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/10 pb-4">
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center shrink-0">
            <FileCheck size={20} className="text-white" />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="text-base font-bold text-white m-0 truncate">
                {metadata?.model_name && metadata.model_name !== "None"
                  ? metadata.model_name
                  : pthPath
                    ? fileBasename(pthPath).replace(/\.(pth|onnx)$/i, "")
                    : t("Model Checkpoint")}
              </h3>
              {metadata && (
                <span className="badge done text-[10px] px-2 py-0.5 font-medium">
                  {t("Valid Checkpoint")}
                </span>
              )}
            </div>
            <p className="text-xs text-neutral-400 m-0 mt-0.5 truncate">
              {pthPath ? pthPath : t("Checkpoint architecture metadata")}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              className="ghost h-8 w-8 p-0 rounded-lg flex items-center justify-center text-neutral-400 hover:text-white"
              aria-label={t("Close")}
            >
              <X size={16} />
            </button>
          )}
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="py-6 space-y-4 text-center">
          <div className="loader" role="progressbar" aria-label={t("Inspecting checkpoint…")}>
            <div className="loaderBar" />
          </div>
          <p className="text-xs text-neutral-400 font-medium">{t("Inspecting checkpoint weights and architecture…")}</p>
        </div>
      )}

      {/* Error state */}
      {error && (
        <div
          role="alert"
          className="p-4 rounded-xl border border-red-500/30 text-red-400 bg-red-500/10 text-xs leading-relaxed"
        >
          {error}
        </div>
      )}

      {/* Loaded metadata */}
      {!loading && !error && metadata && (
        <>
          {/* Top summary row */}
          <div className="flex items-center gap-3 text-xs text-neutral-400 flex-wrap">
            <span className="flex items-center gap-1.5 bg-white/5 border border-white/5 px-2.5 py-1 rounded-lg">
              <User size={13} className="text-white shrink-0" />
              <strong className="text-white font-medium">
                {metadata.author && metadata.author !== "None" ? metadata.author : t("Unknown Creator")}
              </strong>
            </span>
            {metadata.creation_date && metadata.creation_date !== "None" && (
              <span className="flex items-center gap-1.5 bg-white/5 border border-white/5 px-2.5 py-1 rounded-lg">
                <Calendar size={13} className="text-white shrink-0" />
                <span>{metadata.creation_date}</span>
              </span>
            )}
          </div>

          {/* Stat cards grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            <div className="bg-black/30 p-3 rounded-xl border border-white/5 space-y-1">
              <span className="text-neutral-400 text-xs block">{t("Epochs")}</span>
              <span className="text-sm font-semibold text-white block">
                {metadata.epochs && metadata.epochs !== "None" ? metadata.epochs : "—"}
              </span>
            </div>

            <div className="bg-black/30 p-3 rounded-xl border border-white/5 space-y-1">
              <span className="text-neutral-400 text-xs block">{t("Training Steps")}</span>
              <span className="text-sm font-semibold text-white block">
                {metadata.step && metadata.step !== "None" ? Number(metadata.step).toLocaleString() : "—"}
              </span>
            </div>

            <div className="bg-black/30 p-3 rounded-xl border border-white/5 space-y-1">
              <span className="text-neutral-400 text-xs block">{t("Sample Rate")}</span>
              <span className="text-sm font-semibold text-white block">
                {metadata.sr && metadata.sr !== "None" ? `${Number(metadata.sr) / 1000} kHz` : "—"}
              </span>
            </div>

            <div className="bg-black/30 p-3 rounded-xl border border-white/5 space-y-1">
              <span className="text-neutral-400 text-xs block">{t("Pitch Guidance (F0)")}</span>
              <span className="text-sm font-semibold text-white block">
                {metadata.f0 === "1" || metadata.f0 === "True" || metadata.f0 === "true"
                  ? t("Yes")
                  : metadata.f0 === "0" || metadata.f0 === "False" || metadata.f0 === "false"
                    ? t("No")
                    : metadata.f0 || "—"}
              </span>
            </div>

            <div className="bg-black/30 p-3 rounded-xl border border-white/5 space-y-1">
              <span className="text-neutral-400 text-xs block">{t("Vocoder")}</span>
              <span className="text-sm font-semibold text-white block">
                {metadata.vocoder && metadata.vocoder !== "None" ? metadata.vocoder : "HiFi-GAN"}
              </span>
            </div>

            <div className="bg-black/30 p-3 rounded-xl border border-white/5 space-y-1">
              <span className="text-neutral-400 text-xs block">{t("Feature Embedder")}</span>
              <span className="text-sm font-semibold text-white block">
                {metadata.embedder_model && metadata.embedder_model !== "None"
                  ? metadata.embedder_model
                  : "contentvec"}
              </span>
            </div>

            <div className="bg-black/30 p-3 rounded-xl border border-white/5 space-y-1">
              <span className="text-neutral-400 text-xs block">{t("Dataset Slices")}</span>
              <span className="text-sm font-semibold text-white block">
                {metadata.dataset_length && metadata.dataset_length !== "None"
                  ? metadata.dataset_length
                  : "—"}
              </span>
            </div>

            <div className="bg-black/30 p-3 rounded-xl border border-white/5 space-y-1">
              <span className="text-neutral-400 text-xs block">{t("Speakers ID")}</span>
              <span className="text-sm font-semibold text-white block">
                {metadata.speakers_id !== undefined ? String(metadata.speakers_id) : "0"}
              </span>
            </div>
          </div>

          {/* Model Hash pill */}
          {metadata.model_hash && metadata.model_hash !== "None" && (
            <div className="flex items-center justify-between p-3 rounded-xl bg-black/40 border border-white/5 gap-3">
              <div className="flex items-center gap-2 min-w-0">
                <Hash size={15} className="text-white shrink-0" />
                <span className="text-xs text-neutral-400 shrink-0">{t("SHA Hash:")}</span>
                <span className="text-xs text-neutral-200 font-medium truncate select-all">
                  {metadata.model_hash}
                </span>
              </div>
              <button
                type="button"
                className="ghost h-7 px-2.5 rounded-lg flex items-center gap-1.5 text-xs text-neutral-300 hover:text-white shrink-0"
                onClick={() => copyHash(metadata.model_hash || "")}
                aria-label={t("Copy model hash")}
              >
                {copied ? <Check size={13} className="text-white" /> : <Copy size={13} />}
                <span>{copied ? t("Copied") : t("Copy")}</span>
              </button>
            </div>
          )}

          {/* Action Bar */}
          <div className="flex items-center justify-end gap-3 pt-2 border-t border-white/5">
            {onClose && (
              <button
                type="button"
                className="ghost h-10 px-4 text-xs font-medium rounded-xl"
                onClick={onClose}
              >
                {t("Close")}
              </button>
            )}
            <button
              type="button"
              className="cta h-10 px-4 flex items-center gap-2 text-sm font-medium rounded-xl"
              onClick={handleUseInInference}
            >
              <span>{t("Use in Inference")}</span>
              <ArrowRight size={15} className="shrink-0" />
            </button>
          </div>
        </>
      )}
    </div>
  );
}
