"use client";

import { Download, ExternalLink, Pause, Play, Repeat, RotateCcw, Volume2, VolumeX } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useI18n } from "../lib/i18n";

interface AudioPlayerProps {
  src: string;
  originalSrc?: string | null;
  title?: string;
  filename?: string;
  showAnalyzerLink?: boolean;
}

function formatTime(seconds: number): string {
  if (Number.isNaN(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s < 10 ? "0" : ""}${s}`;
}

export default function AudioPlayer({
  src,
  originalSrc,
  title,
  filename,
  showAnalyzerLink = true,
}: AudioPlayerProps) {
  const { t } = useI18n();
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // When A/B compare is available, activeTrack can be 'converted' or 'original'
  const [activeTrack, setActiveTrack] = useState<"converted" | "original">("converted");
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [isLooping, setIsLooping] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [announcement, setAnnouncement] = useState("");

  const currentSrc = activeTrack === "original" && originalSrc ? originalSrc : src;

  // Announce status changes to screen readers
  const announce = (msg: string) => {
    setAnnouncement(msg);
  };

  // Sync state when src changes
  // biome-ignore lint/correctness/useExhaustiveDependencies: reset player state on src prop change
  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
  }, [src]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      announce(t("Audio paused"));
    } else {
      audioRef.current.play().catch(() => {});
      announce(t("Audio playing"));
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = Number(e.target.value);
    setCurrentTime(time);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
    }
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
    announce(next ? t("Audio muted") : t("Audio unmuted"));
  };

  const toggleLoop = () => {
    const next = !isLooping;
    setIsLooping(next);
    if (audioRef.current) {
      audioRef.current.loop = next;
    }
    announce(next ? t("Looping enabled") : t("Looping disabled"));
  };

  const cycleRate = () => {
    const rates = [1, 1.25, 1.5, 2, 0.5, 0.75];
    const nextIndex = (rates.indexOf(playbackRate) + 1) % rates.length;
    const nextRate = rates[nextIndex];
    setPlaybackRate(nextRate);
    if (audioRef.current) {
      audioRef.current.playbackRate = nextRate;
    }
    announce(`${t("Playback speed")} ${nextRate}x`);
  };

  const handleTrackSwitch = (track: "converted" | "original") => {
    if (track === activeTrack) return;
    const savedTime = audioRef.current ? audioRef.current.currentTime : 0;
    const wasPlaying = isPlaying;
    setActiveTrack(track);
    announce(
      track === "original"
        ? t("Switched to original audio track")
        : t("Switched to converted voice model audio"),
    );
    // Give audio element time to switch src then restore time & play state
    setTimeout(() => {
      if (audioRef.current) {
        audioRef.current.currentTime = savedTime;
        if (wasPlaying) {
          audioRef.current.play().catch(() => {});
        }
      }
    }, 50);
  };

  const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0;
  const displayName = filename || (src.split(/[\\/]/).pop() ?? "audio.wav");

  return (
    <section
      aria-label={`Audio Player: ${title || displayName}`}
      className="w-full bg-[#171717]/95 border border-white/15 rounded-2xl p-4 my-3 backdrop-blur-md shadow-xl space-y-3"
    >
      {/* Screen Reader polite status announcements */}
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {announcement}
      </div>

      {/* Hidden audio element */}
      {/* biome-ignore lint/a11y/useMediaCaption: User-converted audio stream has no captions */}
      <audio
        ref={audioRef}
        src={currentSrc}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
      />

      {/* Header with Title & A/B Compare switch */}
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="min-w-0 flex-1">
          <p className="text-sm font-semibold text-white truncate m-0">{title || displayName}</p>
          <p className="text-xs text-neutral-400 truncate m-0">
            {activeTrack === "original" ? t("Original Audio") : t("Converted Voice Model")}
          </p>
        </div>

        {/* A/B Switch if original audio is provided */}
        {originalSrc && (
          <div
            role="tablist"
            aria-label={t("Audio track comparison")}
            className="inline-flex rounded-xl border border-white/10 p-0.5 bg-black/40 shadow-inner"
          >
            <button
              type="button"
              role="tab"
              aria-selected={activeTrack === "original"}
              aria-label={t("Play original audio track")}
              onClick={() => handleTrackSwitch("original")}
              className={`px-3 py-1 text-xs font-medium rounded-lg transition-all focus-visible:outline-2 focus-visible:outline-white ${
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
              aria-label={t("Play converted voice model track")}
              onClick={() => handleTrackSwitch("converted")}
              className={`px-3 py-1 text-xs font-medium rounded-lg transition-all focus-visible:outline-2 focus-visible:outline-white ${
                activeTrack === "converted"
                  ? "bg-white text-black font-semibold shadow-sm"
                  : "text-neutral-400 hover:text-white"
              }`}
            >
              {t("Converted (B)")}
            </button>
          </div>
        )}
      </div>

      {/* Scrubber / Progress Bar */}
      <div className="space-y-1.5">
        <div className="relative w-full h-2.5 bg-white/10 rounded-full overflow-hidden cursor-pointer group">
          <div
            className="absolute top-0 left-0 h-full bg-white transition-all rounded-full"
            style={{ width: `${progressPercent}%` }}
          />
          <input
            type="range"
            min={0}
            max={duration || 100}
            step={0.01}
            value={currentTime}
            onChange={handleSeek}
            aria-label={t("Audio seek position")}
            aria-valuemin={0}
            aria-valuemax={Math.round(duration || 100)}
            aria-valuenow={Math.round(currentTime)}
            aria-valuetext={`${formatTime(currentTime)} of ${formatTime(duration)}`}
            className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer"
          />
        </div>
        <div className="flex justify-between text-xs text-neutral-400 tabular-nums">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {/* Control Buttons */}
      <div className="flex items-center justify-between gap-3 flex-wrap pt-2 border-t border-white/5">
        <div className="flex items-center gap-2">
          {/* Play / Pause button */}
          <button
            type="button"
            onClick={togglePlay}
            aria-label={isPlaying ? t("Pause audio (Space)") : t("Play audio (Space)")}
            className="w-10 h-10 rounded-full bg-white text-black flex items-center justify-center hover:bg-neutral-200 shadow-md focus-visible:outline-2 focus-visible:outline-white focus-visible:outline-offset-2"
            title={isPlaying ? t("Pause (Space)") : t("Play (Space)")}
          >
            {isPlaying ? (
              <Pause size={18} fill="currentColor" aria-hidden="true" />
            ) : (
              <Play size={18} fill="currentColor" className="ml-0.5" aria-hidden="true" />
            )}
          </button>

          {/* Reset button */}
          <button
            type="button"
            onClick={() => {
              if (audioRef.current) audioRef.current.currentTime = 0;
            }}
            aria-label={t("Restart audio")}
            className="p-2 text-neutral-400 hover:text-white transition-colors rounded-lg focus-visible:outline-2 focus-visible:outline-white"
            title={t("Restart")}
          >
            <RotateCcw size={16} aria-hidden="true" />
          </button>

          {/* Loop button */}
          <button
            type="button"
            onClick={toggleLoop}
            aria-label={t("Toggle loop")}
            aria-pressed={isLooping}
            className={`p-2 rounded-lg transition-colors focus-visible:outline-2 focus-visible:outline-white ${
              isLooping ? "text-white bg-white/15" : "text-neutral-400 hover:text-white"
            }`}
            title={isLooping ? t("Looping Enabled") : t("Enable Loop")}
          >
            <Repeat size={16} aria-hidden="true" />
          </button>

          {/* Playback speed toggle */}
          <button
            type="button"
            onClick={cycleRate}
            aria-label={`${t("Playback speed")}: ${playbackRate}x`}
            className="px-2.5 py-1 text-xs font-medium rounded-lg bg-white/5 text-neutral-300 hover:text-white hover:bg-white/10 transition-colors focus-visible:outline-2 focus-visible:outline-white"
            title={t("Playback Speed")}
          >
            {playbackRate}x
          </button>
        </div>

        {/* Volume & Actions */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={toggleMute}
              aria-label={isMuted ? t("Unmute audio") : t("Mute audio")}
              aria-pressed={isMuted}
              className="p-1.5 text-neutral-400 hover:text-white transition-colors rounded-lg focus-visible:outline-2 focus-visible:outline-white"
              title={isMuted ? t("Unmute") : t("Mute")}
            >
              {isMuted || volume === 0 ? (
                <VolumeX size={16} aria-hidden="true" />
              ) : (
                <Volume2 size={16} aria-hidden="true" />
              )}
            </button>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={isMuted ? 0 : volume}
              onChange={handleVolumeChange}
              aria-label={t("Audio volume")}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round((isMuted ? 0 : volume) * 100)}
              aria-valuetext={`${Math.round((isMuted ? 0 : volume) * 100)}%`}
              className="w-16 h-1 bg-white/20 rounded-lg accent-white cursor-pointer"
            />
          </div>

          <a
            href={currentSrc}
            download={displayName}
            aria-label={`${t("Download audio")}: ${displayName}`}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-white/10 hover:bg-white/20 text-white transition-colors focus-visible:outline-2 focus-visible:outline-white"
          >
            <Download size={14} aria-hidden="true" />
            <span>{t("Download")}</span>
          </a>

          {showAnalyzerLink && (
            <Link
              href="/extra"
              aria-label={t("Inspect in Audio Tools")}
              className="p-1.5 text-neutral-400 hover:text-white transition-colors rounded-lg focus-visible:outline-2 focus-visible:outline-white"
              title={t("Inspect in Audio Tools")}
            >
              <ExternalLink size={14} aria-hidden="true" />
            </Link>
          )}
        </div>
      </div>
    </section>
  );
}
