"use client";

import {
  Download,
  ExternalLink,
  Pause,
  Play,
  Repeat,
  RotateCcw,
  Trash2,
  Volume2,
  VolumeX,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { useI18n } from "../lib/i18n";

export interface AudioWavePlayerProps {
  src: string;
  file?: File | null;
  originalSrc?: string | null;
  title?: string;
  filename?: string;
  showAnalyzerLink?: boolean;
  onRemove?: () => void;
  className?: string;
}

function formatTime(seconds: number): string {
  if (Number.isNaN(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s < 10 ? "0" : ""}${s}`;
}

// Generate fallback visual peaks when Web Audio API decoding is pending or unavailable
function generateSyntheticPeaks(count: number, seedStr = "applio"): number[] {
  let hash = 0;
  for (let i = 0; i < seedStr.length; i++) {
    hash = (hash << 5) - hash + seedStr.charCodeAt(i);
    hash |= 0;
  }
  const peaks: number[] = [];
  for (let i = 0; i < count; i++) {
    const x = Math.sin(i * 0.15 + hash * 0.01) * 0.4 + Math.cos(i * 0.35) * 0.3 + 0.45;
    peaks.push(Math.max(0.12, Math.min(0.95, x)));
  }
  return peaks;
}

export default function AudioWavePlayer({
  src,
  file,
  originalSrc,
  title,
  filename,
  showAnalyzerLink = true,
  onRemove,
  className = "",
}: AudioWavePlayerProps) {
  const { t } = useI18n();
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [activeTrack, setActiveTrack] = useState<"converted" | "original">("converted");
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [isLooping, setIsLooping] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [hoverTime, setHoverTime] = useState<number | null>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const [peaks, setPeaks] = useState<number[]>(() =>
    generateSyntheticPeaks(80, title || filename || "audio"),
  );
  const [isScrubbing, setIsScrubbing] = useState(false);

  const currentSrc = activeTrack === "original" && originalSrc ? originalSrc : src;

  // 1. Decode real waveform peaks using Web Audio API
  useEffect(() => {
    let active = true;
    const NUM_BARS = 96;

    async function extractPeaks() {
      try {
        let arrayBuffer: ArrayBuffer;
        if (file && activeTrack !== "original") {
          arrayBuffer = await file.arrayBuffer();
        } else {
          const resp = await fetch(currentSrc);
          if (!resp.ok) return;
          arrayBuffer = await resp.arrayBuffer();
        }

        const AudioCtx =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        if (!AudioCtx) return;
        const ctx = new AudioCtx();
        const decoded = await ctx.decodeAudioData(arrayBuffer);
        if (!active) {
          ctx.close().catch(() => {});
          return;
        }

        const channelData = decoded.getChannelData(0);
        const step = Math.floor(channelData.length / NUM_BARS);
        const newPeaks: number[] = [];

        for (let i = 0; i < NUM_BARS; i++) {
          const start = i * step;
          let maxVal = 0;
          for (let j = 0; j < step; j += 4) {
            const val = Math.abs(channelData[start + j] || 0);
            if (val > maxVal) maxVal = val;
          }
          newPeaks.push(Math.max(0.08, Math.min(1.0, maxVal)));
        }

        // Normalize peaks so waveform looks crisp and full
        const maxPeak = Math.max(...newPeaks, 0.1);
        const normalized = newPeaks.map((p) => Math.max(0.1, p / maxPeak));

        if (active) {
          setPeaks(normalized);
        }
        ctx.close().catch(() => {});
      } catch {
        // Fallback to synthetic peaks if decoding isn't supported for this format
        if (active) {
          setPeaks(generateSyntheticPeaks(NUM_BARS, currentSrc));
        }
      }
    }

    extractPeaks();
    return () => {
      active = false;
    };
  }, [currentSrc, file, activeTrack]);

  // 2. Draw Waveform on Canvas with high-DPI support
  const drawWaveform = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const barCount = peaks.length;
    if (barCount === 0) {
      ctx.restore();
      return;
    }

    const gap = 2.5;
    const barWidth = Math.max(2, (width - (barCount - 1) * gap) / barCount);
    const progressRatio = duration > 0 ? Math.min(1, Math.max(0, currentTime / duration)) : 0;
    const hoverRatio = hoverTime !== null && duration > 0 ? hoverTime / duration : null;

    const centerY = height / 2;

    for (let i = 0; i < barCount; i++) {
      const x = i * (barWidth + gap);
      const barRatio = i / barCount;
      const peak = peaks[i];
      const barHeight = Math.max(4, peak * (height - 8));
      const topY = centerY - barHeight / 2;

      const isPlayed = barRatio <= progressRatio;
      const isHovered = hoverRatio !== null && barRatio <= hoverRatio;

      if (isPlayed) {
        // Played portion: bright white with subtle glow
        ctx.fillStyle = "#ffffff";
      } else if (isHovered) {
        // Hover scrub trail: semi-bright
        ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
      } else {
        // Unplayed portion: dark muted gray
        ctx.fillStyle = "rgba(255, 255, 255, 0.18)";
      }

      // Draw rounded bar
      ctx.beginPath();
      const radius = barWidth / 2;
      ctx.roundRect(x, topY, barWidth, barHeight, radius);
      ctx.fill();
    }

    // Draw playhead cursor line
    const playheadX = progressRatio * width;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(playheadX, centerY, 3.5, 0, Math.PI * 2);
    ctx.fill();

    // Draw hover cursor line if hovering
    if (hoverX !== null) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hoverX, 2);
      ctx.lineTo(hoverX, height - 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.restore();
  }, [peaks, currentTime, duration, hoverTime, hoverX]);

  useEffect(() => {
    drawWaveform();
  }, [drawWaveform]);

  // Window resize re-draw
  useEffect(() => {
    const handleResize = () => drawWaveform();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [drawWaveform]);

  // Sync state when src changes
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset player state on src change
  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
  }, [src]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play().catch(() => {});
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current && !isScrubbing) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration || 0);
    }
  };

  const seekFromMouseEvent = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !audioRef.current || duration <= 0) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const ratio = x / rect.width;
    const newTime = ratio * duration;
    setCurrentTime(newTime);
    audioRef.current.currentTime = newTime;
  };

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsScrubbing(true);
    seekFromMouseEvent(e);
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || duration <= 0) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const ratio = x / rect.width;
    setHoverX(x);
    setHoverTime(ratio * duration);
    if (isScrubbing) {
      seekFromMouseEvent(e);
    }
  };

  const handleCanvasMouseLeave = () => {
    setHoverX(null);
    setHoverTime(null);
    setIsScrubbing(false);
  };

  const handleCanvasMouseUp = () => {
    setIsScrubbing(false);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = Number(e.target.value);
    setVolume(val);
    setIsMuted(val === 0);
    if (audioRef.current) {
      audioRef.current.volume = val;
      audioRef.current.muted = val === 0;
    }
  };

  const toggleMute = () => {
    if (!audioRef.current) return;
    const next = !isMuted;
    setIsMuted(next);
    audioRef.current.muted = next;
  };

  const toggleLoop = () => {
    const next = !isLooping;
    setIsLooping(next);
    if (audioRef.current) {
      audioRef.current.loop = next;
    }
  };

  const cycleRate = () => {
    const rates = [1, 1.25, 1.5, 2, 0.75];
    const nextIndex = (rates.indexOf(playbackRate) + 1) % rates.length;
    const nextRate = rates[nextIndex];
    setPlaybackRate(nextRate);
    if (audioRef.current) {
      audioRef.current.playbackRate = nextRate;
    }
  };

  const handleTrackSwitch = (track: "converted" | "original") => {
    if (track === activeTrack) return;
    const savedTime = audioRef.current ? audioRef.current.currentTime : 0;
    const wasPlaying = isPlaying;
    setActiveTrack(track);

    setTimeout(() => {
      if (audioRef.current) {
        audioRef.current.currentTime = savedTime;
        if (wasPlaying) {
          audioRef.current.play().catch(() => {});
        }
      }
    }, 50);
  };

  const displayName = filename || (src.split(/[\\/]/).pop() ?? "audio.wav");

  return (
    <section
      aria-label={`Audio Waveplayer: ${title || displayName}`}
      className={`w-full bg-[#141414] border border-white/10 rounded-2xl p-4 my-2 backdrop-blur-md shadow-2xl space-y-3 transition-all ${className}`}
    >
      {/* Audio element */}
      {/* biome-ignore lint/a11y/useMediaCaption: Audio waveform stream has no captions */}
      <audio
        ref={audioRef}
        src={currentSrc}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
      />

      {/* Header: Title, Metadata, A/B compare switch & optional delete */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-white/80 shrink-0" />
            <p className="text-sm font-semibold text-white truncate m-0">{title || displayName}</p>
          </div>
          <p className="text-xs text-neutral-400 truncate m-0 ml-4">
            {activeTrack === "original" ? t("Original Audio Track") : t("Converted Voice Model Track")}
          </p>
        </div>

        {/* A/B Compare Switch */}
        {originalSrc && (
          <div
            role="tablist"
            aria-label={t("Audio comparison")}
            className="inline-flex rounded-xl border border-white/10 p-0.5 bg-black/60 shadow-inner shrink-0"
          >
            <button
              type="button"
              role="tab"
              aria-selected={activeTrack === "original"}
              onClick={() => handleTrackSwitch("original")}
              className={`px-3 py-1 text-xs font-medium rounded-lg transition-all ${
                activeTrack === "original"
                  ? "bg-white/20 text-white font-semibold shadow-sm"
                  : "text-neutral-400 hover:text-white"
              }`}
            >
              {t("Original (A)")}
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTrack === "converted"}
              onClick={() => handleTrackSwitch("converted")}
              className={`px-3 py-1 text-xs font-medium rounded-lg transition-all ${
                activeTrack === "converted"
                  ? "bg-white text-black font-semibold shadow-sm"
                  : "text-neutral-400 hover:text-white"
              }`}
            >
              {t("Converted (B)")}
            </button>
          </div>
        )}

        {/* Remove button if passed */}
        {onRemove && (
          <button
            type="button"
            onClick={onRemove}
            aria-label={t("Remove audio")}
            title={t("Remove audio")}
            className="p-1.5 text-neutral-400 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors shrink-0"
          >
            <Trash2 size={16} />
          </button>
        )}
      </div>

      {/* Waveform Canvas Container with Hover Scrub Line */}
      <div
        ref={containerRef}
        className="relative w-full h-16 bg-black/40 border border-white/5 rounded-xl overflow-hidden cursor-pointer select-none group"
      >
        <canvas
          ref={canvasRef}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleCanvasMouseMove}
          onMouseLeave={handleCanvasMouseLeave}
          onMouseUp={handleCanvasMouseUp}
          className="w-full h-full block"
        />

        {/* Hover Time Tooltip */}
        {hoverTime !== null && hoverX !== null && (
          <div
            className="absolute top-1.5 px-2 py-0.5 rounded bg-black/90 text-[10px] font-mono font-medium text-white pointer-events-none transform -translate-x-1/2 shadow-md border border-white/20 z-10"
            style={{ left: hoverX }}
          >
            {formatTime(hoverTime)}
          </div>
        )}
      </div>

      {/* Time Display and Wave Controls Bar */}
      <div className="flex items-center justify-between gap-3 flex-wrap pt-1">
        {/* Left: Play/Pause, Restart, Loop, Speed */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={togglePlay}
            aria-label={isPlaying ? t("Pause audio") : t("Play audio")}
            className="w-9 h-9 rounded-full bg-white text-black flex items-center justify-center hover:bg-neutral-200 shadow-md transition-colors focus-visible:outline-2 focus-visible:outline-white shrink-0"
          >
            {isPlaying ? (
              <Pause size={16} fill="currentColor" />
            ) : (
              <Play size={16} fill="currentColor" className="ml-0.5" />
            )}
          </button>

          <button
            type="button"
            onClick={() => {
              if (audioRef.current) audioRef.current.currentTime = 0;
            }}
            aria-label={t("Restart audio")}
            title={t("Restart")}
            className="p-2 text-neutral-400 hover:text-white rounded-lg transition-colors"
          >
            <RotateCcw size={15} />
          </button>

          <button
            type="button"
            onClick={toggleLoop}
            aria-label={t("Toggle loop")}
            aria-pressed={isLooping}
            title={isLooping ? t("Looping Enabled") : t("Enable Loop")}
            className={`p-2 rounded-lg transition-colors ${
              isLooping ? "text-white bg-white/15" : "text-neutral-400 hover:text-white"
            }`}
          >
            <Repeat size={15} />
          </button>

          <button
            type="button"
            onClick={cycleRate}
            title={t("Playback Speed")}
            className="px-2 py-1 text-xs font-mono font-medium rounded-lg bg-white/5 text-neutral-300 hover:text-white hover:bg-white/10 transition-colors"
          >
            {playbackRate}x
          </button>

          <div className="text-xs text-neutral-400 tabular-nums font-mono ml-1">
            <span className="text-white font-medium">{formatTime(currentTime)}</span>
            <span className="mx-1">/</span>
            <span>{formatTime(duration)}</span>
          </div>
        </div>

        {/* Right: Volume & Actions */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={toggleMute}
              aria-label={isMuted ? t("Unmute audio") : t("Mute audio")}
              className="p-1.5 text-neutral-400 hover:text-white transition-colors"
            >
              {isMuted || volume === 0 ? <VolumeX size={15} /> : <Volume2 size={15} />}
            </button>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={isMuted ? 0 : volume}
              onChange={handleVolumeChange}
              aria-label={t("Volume")}
              className="w-16 h-1 bg-white/20 rounded-lg accent-white cursor-pointer"
            />
          </div>

          <a
            href={currentSrc}
            download={displayName}
            aria-label={`${t("Download audio")}: ${displayName}`}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-white/10 hover:bg-white/20 text-white transition-colors"
          >
            <Download size={13} />
            <span>{t("Download")}</span>
          </a>

          {showAnalyzerLink && (
            <Link
              href="/extra"
              aria-label={t("Inspect in Audio Tools")}
              title={t("Inspect in Audio Tools")}
              className="p-1.5 text-neutral-400 hover:text-white transition-colors rounded-lg"
            >
              <ExternalLink size={14} />
            </Link>
          )}
        </div>
      </div>
    </section>
  );
}
