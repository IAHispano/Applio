"use client";

import { FileAudio, Mic, Music, UploadCloud } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useI18n } from "../../lib/i18n";
import AudioWavePlayer from "../AudioWavePlayer";

export interface AudioDropzoneProps {
  audioFile: File | null;
  inputPath: string;
  sampleAudios: string[];
  onFileSelect: (file: File | null) => void;
  onPathSelect: (path: string) => void;
  disabled?: boolean;
}

function pickMime(): string {
  if (typeof MediaRecorder === "undefined" || !MediaRecorder.isTypeSupported) return "";
  for (const m of ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"]) {
    try {
      if (MediaRecorder.isTypeSupported(m)) return m;
    } catch {
      /* ignore */
    }
  }
  return "";
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${Number.parseFloat((bytes / k ** i).toFixed(1))} ${sizes[i]}`;
}

export default function AudioDropzone({
  audioFile,
  inputPath,
  sampleAudios,
  onFileSelect,
  onPathSelect,
  disabled = false,
}: AudioDropzoneProps) {
  const { t } = useI18n();
  const [tab, setTab] = useState<"upload" | "samples" | "mic">("upload");
  const [isDragging, setIsDragging] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recDuration, setRecDuration] = useState(0);
  const [micError, setMicError] = useState("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const recRef = useRef<{ rec: MediaRecorder; chunks: Blob[]; stream: MediaStream } | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const hasAudio = !!audioFile || !!inputPath;

  // Cleanup mic recording on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      recRef.current?.stream.getTracks().forEach((t) => {
        t.stop();
      });
    };
  }, []);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (disabled) return;

    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles && droppedFiles.length > 0) {
      const file = droppedFiles[0];
      if (file.type.startsWith("audio/") || /\.(wav|mp3|flac|ogg|m4a|opus|webm|aac|aiff)$/i.test(file.name)) {
        onFileSelect(file);
        onPathSelect("");
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    if (file) {
      onFileSelect(file);
      onPathSelect("");
    }
  };

  const startMic = async () => {
    setMicError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = pickMime();
      const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      const chunks: Blob[] = [];

      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      rec.onstop = () => {
        stream.getTracks().forEach((t) => {
          t.stop();
        });
        setRecording(false);
        if (timerRef.current) clearInterval(timerRef.current);

        const blob = new Blob(chunks, { type: mime || "audio/webm" });
        const file = new File([blob], `mic-recording-${Date.now()}.webm`, {
          type: blob.type,
        });
        onFileSelect(file);
        onPathSelect("");
      };

      recRef.current = { rec, chunks, stream };
      rec.start();
      setRecording(true);
      setRecDuration(0);

      timerRef.current = setInterval(() => {
        setRecDuration((d) => d + 1);
      }, 1000);
    } catch {
      setMicError(t("Microphone unavailable — check permissions."));
      setRecording(false);
    }
  };

  const stopMic = () => {
    recRef.current?.rec.stop();
  };

  const clearAudio = () => {
    onFileSelect(null);
    onPathSelect("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  // Preview URL for the waveplayer
  const previewSrc = audioFile ? URL.createObjectURL(audioFile) : inputPath ? `/${inputPath}` : "";

  return (
    <div className="w-full space-y-3">
      {/* Tab Switcher: Upload / Samples / Mic */}
      <div className="flex items-center justify-between border-b border-white/10 pb-2">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setTab("upload")}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${
              tab === "upload"
                ? "bg-white text-black font-semibold shadow-sm"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            <span className="flex items-center gap-1.5">
              <UploadCloud size={14} />
              <span>{t("Upload / Drag & Drop")}</span>
            </span>
          </button>

          <button
            type="button"
            onClick={() => setTab("samples")}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${
              tab === "samples"
                ? "bg-white text-black font-semibold shadow-sm"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            <span className="flex items-center gap-1.5">
              <Music size={14} />
              <span>{t("Sample Library")}</span>
            </span>
          </button>

          <button
            type="button"
            onClick={() => setTab("mic")}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${
              tab === "mic"
                ? "bg-white text-black font-semibold shadow-sm"
                : "text-neutral-400 hover:text-white"
            }`}
          >
            <span className="flex items-center gap-1.5">
              <Mic size={14} />
              <span>{t("Microphone")}</span>
            </span>
          </button>
        </div>

        {hasAudio && (
          <button
            type="button"
            onClick={clearAudio}
            className="text-xs text-neutral-400 hover:text-red-400 transition-colors"
          >
            {t("Clear audio")}
          </button>
        )}
      </div>

      {/* When audio is active, show the AudioWavePlayer preview */}
      {hasAudio ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-neutral-400 px-1">
            <span>{t("Active Input Audio Preview")}</span>
            {audioFile && <span>{formatBytes(audioFile.size)}</span>}
          </div>

          <AudioWavePlayer
            src={previewSrc}
            file={audioFile}
            title={audioFile ? audioFile.name : inputPath.split(/[\\/]/).pop() || inputPath}
            showAnalyzerLink={false}
            onRemove={clearAudio}
          />
        </div>
      ) : (
        /* When no audio is selected, render tab contents */
        <div>
          {tab === "upload" && (
            // biome-ignore lint/a11y/useSemanticElements: interactive dropzone container
            <div
              onDragOver={handleDragOver}
              onDragEnter={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  fileInputRef.current?.click();
                }
              }}
              role="button"
              tabIndex={0}
              className={`relative w-full py-8 px-6 rounded-2xl border-2 border-dashed transition-all cursor-pointer flex flex-col items-center justify-center text-center select-none ${
                isDragging
                  ? "border-white bg-white/15 scale-[1.01] shadow-2xl"
                  : "border-white/15 bg-white/[0.03] hover:border-white/30 hover:bg-white/[0.06]"
              } ${disabled ? "opacity-40 cursor-not-allowed" : ""}`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".wav,.mp3,.flac,.ogg,.opus,.m4a,.mp4,.aac,.aiff,.webm"
                onChange={handleFileChange}
                disabled={disabled}
                className="hidden"
              />

              <div className="w-12 h-12 rounded-2xl bg-white/10 flex items-center justify-center text-white mb-3 shadow-inner">
                <UploadCloud size={24} />
              </div>

              <p className="text-sm font-semibold text-white m-0">
                {isDragging ? t("Drop audio file here") : t("Click to browse or drag and drop audio")}
              </p>
              <p className="text-xs text-neutral-400 m-0 mt-1 max-w-sm">
                {t("Supports WAV, MP3, FLAC, OGG, M4A, Opus, and AAC (max 200MB)")}
              </p>

              <div className="flex items-center gap-1.5 mt-4 flex-wrap justify-center">
                {["WAV", "MP3", "FLAC", "OGG", "M4A"].map((fmt) => (
                  <span
                    key={fmt}
                    className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-full bg-white/10 text-neutral-300 border border-white/5"
                  >
                    {fmt}
                  </span>
                ))}
              </div>
            </div>
          )}

          {tab === "samples" && (
            <div className="p-3 bg-white/[0.03] border border-white/10 rounded-2xl space-y-2">
              <label htmlFor="sample-audio-select" className="text-xs font-medium text-neutral-300">
                {t("Pick a sample audio from assets/audios")}
              </label>
              {sampleAudios.length === 0 ? (
                <p className="text-xs text-neutral-400 m-0 py-2">
                  {t("No sample files found in assets/audios")}
                </p>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-48 overflow-y-auto">
                  {sampleAudios.map((s) => {
                    const name = s.split(/[\\/]/).pop() || s;
                    return (
                      <button
                        key={s}
                        type="button"
                        onClick={() => {
                          onPathSelect(s);
                          onFileSelect(null);
                        }}
                        className="flex items-center gap-2.5 p-2 rounded-xl text-left border border-white/5 bg-white/5 hover:bg-white/10 hover:border-white/20 transition-all group"
                      >
                        <FileAudio size={16} className="text-neutral-400 group-hover:text-white shrink-0" />
                        <span className="text-xs font-medium text-neutral-200 group-hover:text-white truncate">
                          {name}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {tab === "mic" && (
            <div className="p-6 bg-white/[0.03] border border-white/10 rounded-2xl flex flex-col items-center justify-center text-center space-y-4">
              <div
                className={`w-16 h-16 rounded-full flex items-center justify-center transition-all ${
                  recording
                    ? "bg-red-500 text-white animate-pulse shadow-lg shadow-red-500/40 scale-110"
                    : "bg-white/10 text-white hover:bg-white/20"
                }`}
              >
                <Mic size={28} />
              </div>

              <div>
                <p className="text-sm font-semibold text-white m-0">
                  {recording ? t("Recording in progress…") : t("Record directly with microphone")}
                </p>
                <p className="text-xs font-mono text-neutral-400 m-0 mt-1">
                  {recording
                    ? `${Math.floor(recDuration / 60)}:${recDuration % 60 < 10 ? "0" : ""}${recDuration % 60}`
                    : t("High quality voice capture")}
                </p>
              </div>

              <div className="flex items-center gap-2">
                {!recording ? (
                  <button type="button" onClick={startMic} className="cta text-xs">
                    {t("Start recording")}
                  </button>
                ) : (
                  <button
                    type="button"
                    onClick={stopMic}
                    className="px-4 py-2 text-xs font-semibold rounded-lg bg-red-600 hover:bg-red-500 text-white transition-colors shadow-md"
                  >
                    {t("Stop recording")}
                  </button>
                )}
              </div>

              {micError && <p className="text-xs text-red-400 m-0">{micError}</p>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
