import {
  Activity,
  Blocks,
  Bug,
  Cpu,
  Database,
  Download,
  House,
  Layers,
  type LucideIcon,
  Mic,
  Radio,
  Settings,
  SlidersHorizontal,
  Sparkles,
} from "lucide-react";

export interface NavEntry {
  icon: LucideIcon;
  label: string;
  to: string;
  blurb: string;
  badge?: string;
}

export interface NavSection {
  title?: string;
  items: NavEntry[];
}

export const NAV_SECTIONS: NavSection[] = [
  {
    items: [{ icon: House, label: "Home", to: "/", blurb: "Overview, quick actions and system status" }],
  },
  {
    title: "Voice Studio",
    items: [
      { icon: Sparkles, label: "Inference", to: "/inference", blurb: "Convert audio files single or batch" },
      { icon: Radio, label: "Realtime", to: "/realtime", blurb: "Live microphone voice conversion" },
      { icon: Mic, label: "TTS", to: "/tts", blurb: "Synthesize speech and convert voice" },
    ],
  },
  {
    title: "Models & Training",
    items: [
      { icon: Database, label: "Models", to: "/models", blurb: "Manage installed voice models and weights" },
      { icon: Cpu, label: "Training", to: "/train", blurb: "Preprocess, extract, train, and index" },
      { icon: Layers, label: "Voice Blender", to: "/voice-blender", blurb: "Fuse two models into a hybrid" },
      { icon: Download, label: "Download", to: "/download", blurb: "Download models and pretrained weights" },
      { icon: Activity, label: "TensorBoard", to: "/tensorboard", blurb: "Track training loss and metrics" },
    ],
  },
  {
    title: "Tools & Extras",
    items: [
      { icon: SlidersHorizontal, label: "Extra", to: "/extra", blurb: "Audio analyzer and pitch extractor" },
      { icon: Blocks, label: "Plugins", to: "/plugins", blurb: "Extend Applio with plugins" },
    ],
  },
  {
    title: "System",
    items: [
      { icon: Settings, label: "Settings", to: "/settings", blurb: "Engine configuration and preferences" },
      { icon: Bug, label: "Report a Bug", to: "/report", blurb: "System diagnostics and issue reporting" },
    ],
  },
];

export const MENU: NavEntry[] = NAV_SECTIONS.flatMap((s) => s.items);
