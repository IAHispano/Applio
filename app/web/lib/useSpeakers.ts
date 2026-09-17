"use client";

import { useEffect, useState } from "react";
import { apiGet } from "./api";

// Speaker IDs for multi-speaker models (Gradio get_speakers_id parity).
// Falls back to [0] while loading or for single-speaker models.
export function useSpeakers(pthPath: string): number[] {
  const [speakers, setSpeakers] = useState<number[]>([0]);
  useEffect(() => {
    if (!pthPath) {
      setSpeakers([0]);
      return;
    }
    let live = true;
    const t = setTimeout(() => {
      apiGet<{ speakers: number[] }>(`/api/models/speakers?pthPath=${encodeURIComponent(pthPath)}`)
        .then((r) => {
          if (live) setSpeakers(r.speakers.length > 0 ? r.speakers : [0]);
        })
        .catch(() => {
          if (live) setSpeakers([0]);
        });
    }, 400);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [pthPath]);
  return speakers;
}
