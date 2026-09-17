"use client";

import { Download, ExternalLink, Pause, Play, Repeat, RotateCcw, Volume2, VolumeX } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

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

  const currentSrc = activeTrack === "original" && originalSrc ? originalSrc : src;

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
    } else {
      audioRef.current.play().catch(() => {});
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
  };

  const toggleLoop = () => {
    const next = !isLooping;
    setIsLooping(next);
    if (audioRef.current) {
      audioRef.current.loop = next;
    }
  };

  const cycleRate = () => {
    const rates = [1, 1.25, 1.5, 2, 0.5, 0.75];
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
  const displayName = filename || (src.split("/").pop() ?? "audio.wav");

  return (
    <div className="w-full bg-neutral-900/90 border border-white/10 rounded-xl p-4 my-3 backdrop-blur-sm shadow-lg space-y-3">
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
            {activeTrack === "original" ? "Original Audio" : "Converted Voice Model"}
          </p>
        </div>

        {/* A/B Switch if original audio is provided */}
        {originalSrc && (
          <div className="inline-flex rounded-lg border border-white/10 p-0.5 bg-black/40">
            <button
              type="button"
              onClick={() => handleTrackSwitch("original")}
              className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all ${
                activeTrack === "original"
                  ? "bg-white/20 text-white shadow-sm"
                  : "text-neutral-400 hover:text-white"
              }`}
            >
              Original (A)
            </button>
            <button
              type="button"
              onClick={() => handleTrackSwitch("converted")}
              className={`px-2.5 py-1 text-xs font-medium rounded-md transition-all ${
                activeTrack === "converted"
                  ? "bg-white text-black font-semibold shadow-sm"
                  : "text-neutral-400 hover:text-white"
              }`}
            >
              Converted (B)
            </button>
          </div>
        )}
      </div>

      {/* Scrubber / Progress Bar */}
      <div className="space-y-1">
        <div className="relative w-full h-2 bg-white/10 rounded-full overflow-hidden cursor-pointer">
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
            className="absolute top-0 left-0 w-full h-full opacity-0 cursor-pointer"
          />
        </div>
        <div className="flex justify-between text-xs text-neutral-400 tabular-nums">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {/* Control Buttons */}
      <div className="flex items-center justify-between gap-3 flex-wrap pt-1 border-t border-white/5">
        <div className="flex items-center gap-2">
          {/* Play / Pause button */}
          <button
            type="button"
            onClick={togglePlay}
            className="w-10 h-10 rounded-full bg-white text-black flex items-center justify-center hover:bg-neutral-200 transition-transform active:scale-95 shadow-md"
            title={isPlaying ? "Pause (Space)" : "Play (Space)"}
          >
            {isPlaying ? (
              <Pause size={18} fill="currentColor" />
            ) : (
              <Play size={18} fill="currentColor" className="ml-0.5" />
            )}
          </button>

          {/* Reset button */}
          <button
            type="button"
            onClick={() => {
              if (audioRef.current) audioRef.current.currentTime = 0;
            }}
            className="p-2 text-neutral-400 hover:text-white transition-colors"
            title="Restart"
          >
            <RotateCcw size={16} />
          </button>

          {/* Loop button */}
          <button
            type="button"
            onClick={toggleLoop}
            className={`p-2 rounded-md transition-colors ${
              isLooping ? "text-white bg-white/10" : "text-neutral-400 hover:text-white"
            }`}
            title={isLooping ? "Looping Enabled" : "Enable Loop"}
          >
            <Repeat size={16} />
          </button>

          {/* Playback speed toggle */}
          <button
            type="button"
            onClick={cycleRate}
            className="px-2 py-1 text-xs font-medium rounded-md bg-white/5 text-neutral-300 hover:text-white hover:bg-white/10 transition-colors"
            title="Playback Speed"
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
              className="p-1 text-neutral-400 hover:text-white transition-colors"
              title={isMuted ? "Unmute" : "Mute"}
            >
              {isMuted || volume === 0 ? <VolumeX size={16} /> : <Volume2 size={16} />}
            </button>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={isMuted ? 0 : volume}
              onChange={handleVolumeChange}
              className="w-16 h-1 bg-white/20 rounded-lg accent-white cursor-pointer"
            />
          </div>

          <a
            href={currentSrc}
            download={displayName}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-white/10 hover:bg-white/20 text-white transition-colors"
          >
            <Download size={14} />
            <span>Download</span>
          </a>

          {showAnalyzerLink && (
            <Link
              href="/extra"
              className="p-1.5 text-neutral-400 hover:text-white transition-colors"
              title="Inspect in Audio Tools"
            >
              <ExternalLink size={14} />
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}
