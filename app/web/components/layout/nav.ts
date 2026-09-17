import {
  Activity,
  Blocks,
  Bug,
  Cpu,
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
}

export const MENU: NavEntry[] = [
  { icon: House, label: "Home", to: "/", blurb: "Status, setup and shortcuts" },
  { icon: Sparkles, label: "Inference", to: "/inference", blurb: "Convert voice, single or batch" },
  { icon: Cpu, label: "Training", to: "/train", blurb: "Preprocess, extract, train, index" },
  { icon: Mic, label: "TTS", to: "/tts", blurb: "Synthesize speech, then convert it" },
  { icon: Layers, label: "Voice Blender", to: "/voice-blender", blurb: "Fuse two models into one" },
  { icon: Radio, label: "Realtime", to: "/realtime", blurb: "Live voice conversion" },
  { icon: Blocks, label: "Plugins", to: "/plugins", blurb: "Extend Applio with plugins" },
  { icon: Download, label: "Download", to: "/download", blurb: "Models and pretraineds" },
  { icon: Bug, label: "Report a Bug", to: "/report", blurb: "Capture and file issues" },
  { icon: SlidersHorizontal, label: "Extra", to: "/extra", blurb: "Analyzer, model info, F0 curves" },
  { icon: Settings, label: "Settings", to: "/settings", blurb: "Languages, precision, versions" },
  { icon: Activity, label: "TensorBoard", to: "/tensorboard", blurb: "Watch training live" },
];
