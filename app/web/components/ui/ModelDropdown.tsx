"use client";

import { Check, ChevronDown, Mic2, RefreshCw, Search, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../../lib/i18n";

export interface ModelDropdownProps {
  models: string[];
  selectedModel: string;
  onSelect: (modelPath: string) => void;
  onRefresh?: () => void;
  onUnload?: () => void;
  indexes?: string[];
  disabled?: boolean;
}

function modelDisplayName(path: string): string {
  if (!path) return "";
  const filename = path.split(/[\\/]/).pop() || path;
  return filename.replace(/\.(pth|onnx)$/i, "");
}

function modelFolder(path: string): string {
  if (!path) return "";
  const parts = path.split(/[\\/]/);
  if (parts.length > 1) {
    return parts.slice(0, -1).join("/");
  }
  return "logs";
}

interface MenuPosition {
  left: number;
  width: number;
  top?: number;
  bottom?: number;
}

export default function ModelDropdown({
  models,
  selectedModel,
  onSelect,
  onRefresh,
  onUnload,
  indexes = [],
  disabled = false,
}: ModelDropdownProps) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [mounted, setMounted] = useState(false);
  const [menuPos, setMenuPos] = useState<MenuPosition | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const searchInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Position the floating menu against the trigger button. Rendered in a
  // portal so ancestor cards (backdrop-filter creates a stacking context)
  // can never paint over it.
  const updatePosition = useCallback(() => {
    const el = triggerRef.current;
    if (!el || typeof window === "undefined") return;
    const rect = el.getBoundingClientRect();
    const gap = 6;
    const spaceBelow = window.innerHeight - rect.bottom;
    const openUp = spaceBelow < 260 && rect.top > spaceBelow;
    const width = Math.max(rect.width, 240);
    const left = Math.max(8, Math.min(rect.left, window.innerWidth - width - 8));
    if (openUp) {
      setMenuPos({
        left,
        width,
        bottom: Math.max(8, window.innerHeight - rect.top + gap),
      });
    } else {
      setMenuPos({
        left,
        width,
        top: Math.min(rect.bottom + gap, window.innerHeight - 120),
      });
    }
  }, []);

  useEffect(() => {
    if (!open) {
      setMenuPos(null);
      return;
    }
    updatePosition();
    setTimeout(() => searchInputRef.current?.focus(), 50);

    function handlePointerDown(e: MouseEvent) {
      const target = e.target as Node;
      if (triggerRef.current?.contains(target)) return;
      if (menuRef.current?.contains(target)) return;
      setOpen(false);
    }
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    function handleReposition() {
      updatePosition();
    }
    document.addEventListener("mousedown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("resize", handleReposition);
    // Capture scrolls from any scrollable ancestor (e.g. the main column).
    document.addEventListener("scroll", handleReposition, true);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("resize", handleReposition);
      document.removeEventListener("scroll", handleReposition, true);
    };
  }, [open, updatePosition]);

  const filteredModels = useMemo(() => {
    if (!search.trim()) return models;
    const query = search.toLowerCase();
    return models.filter((m) => m.toLowerCase().includes(query));
  }, [models, search]);

  const hasIndexMatch = (modelPath: string): boolean => {
    const stem = modelDisplayName(modelPath).toLowerCase().slice(0, 8);
    return indexes.some((idx) => idx.toLowerCase().includes(stem));
  };

  const currentDisplayName = selectedModel ? modelDisplayName(selectedModel) : t("Select a voice model…");

  const menu =
    open && mounted && menuPos && typeof document !== "undefined"
      ? createPortal(
          <div
            ref={menuRef}
            role="listbox"
            style={{
              position: "fixed",
              zIndex: 9999,
              left: menuPos.left,
              width: menuPos.width,
              top: menuPos.top,
              bottom: menuPos.bottom,
            }}
            className="rounded-xl bg-[#171717] border border-white/15 shadow-2xl backdrop-blur-xl overflow-hidden animate-in fade-in slide-in-from-top-2 duration-150 flex flex-col max-h-[min(24rem,calc(100vh-120px))]"
          >
            {/* Search Header */}
            <div className="p-2 border-b border-white/10 flex items-center gap-2 bg-black/40 shrink-0">
              <Search size={14} className="text-neutral-400 ml-1 shrink-0" />
              <input
                ref={searchInputRef}
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder={t("Search models…")}
                className="w-full bg-transparent text-xs text-white placeholder-neutral-500 border-none outline-none py-1 focus:ring-0"
              />
              {search && (
                <button
                  type="button"
                  onClick={() => setSearch("")}
                  className="p-1 text-neutral-400 hover:text-white rounded"
                >
                  <X size={12} />
                </button>
              )}
            </div>

            {/* Model Items List */}
            <div className="max-h-60 overflow-y-auto p-1.5 space-y-1 grow">
              {filteredModels.length === 0 ? (
                <div className="p-4 text-center">
                  <p className="text-xs text-neutral-400 m-0">
                    {search ? t("No models matching search query") : t("No models found in logs/")}
                  </p>
                </div>
              ) : (
                filteredModels.map((m) => {
                  const isSelected = m === selectedModel;
                  const hasIndex = hasIndexMatch(m);
                  return (
                    <button
                      key={m}
                      type="button"
                      role="option"
                      aria-selected={isSelected}
                      onClick={() => {
                        onSelect(m);
                        setOpen(false);
                      }}
                      className={`w-full flex items-center justify-between gap-3 px-3 py-2 rounded-lg text-left transition-all ${
                        isSelected
                          ? "bg-white text-black font-semibold shadow-sm"
                          : "text-neutral-200 hover:bg-white/10"
                      }`}
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-semibold truncate">{modelDisplayName(m)}</span>
                          <span
                            className={`text-[9px] px-1 py-0.2 rounded ${
                              isSelected ? "bg-black/20 text-black" : "bg-white/10 text-neutral-400"
                            }`}
                          >
                            {m.endsWith(".onnx") ? "ONNX" : "PTH"}
                          </span>
                        </div>
                        <p
                          className={`text-[10px] truncate m-0 leading-tight mt-0.5 ${
                            isSelected ? "text-neutral-800" : "text-neutral-400"
                          }`}
                        >
                          {m} {hasIndex ? `• ${t("Index paired")}` : ""}
                        </p>
                      </div>

                      {isSelected && <Check size={14} className="shrink-0 text-black" />}
                    </button>
                  );
                })
              )}
            </div>

            {/* Actions Footer */}
            <div className="p-2 border-t border-white/10 bg-black/40 flex items-center justify-between gap-2 text-xs shrink-0">
              <span className="text-[11px] text-neutral-400">
                {models.length} {t("models available")}
              </span>
              <div className="flex items-center gap-2">
                {selectedModel && onUnload && (
                  <button
                    type="button"
                    onClick={() => {
                      onUnload();
                      setOpen(false);
                    }}
                    className="text-xs text-neutral-400 hover:text-red-400 transition-colors"
                  >
                    {t("Unload")}
                  </button>
                )}
                {onRefresh && (
                  <button
                    type="button"
                    onClick={() => {
                      onRefresh();
                    }}
                    className="flex items-center gap-1 text-xs text-neutral-300 hover:text-white transition-colors"
                  >
                    <RefreshCw size={11} />
                    <span>{t("Refresh")}</span>
                  </button>
                )}
              </div>
            </div>
          </div>,
          document.body,
        )
      : null;

  return (
    <div className="relative w-full">
      {/* Dropdown Trigger Button */}
      <button
        ref={triggerRef}
        type="button"
        disabled={disabled}
        onClick={() => setOpen(!open)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={`w-full flex items-center justify-between gap-3 px-3.5 py-2.5 rounded-xl border text-left transition-all ${
          open
            ? "bg-[#1f1f1f] border-white/40 shadow-lg ring-1 ring-white/30"
            : "bg-white/5 border-white/10 hover:border-white/25 hover:bg-white/[0.08]"
        } ${disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer"}`}
      >
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center text-white shrink-0">
            <Mic2 size={16} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span
                className={`text-sm font-semibold truncate ${selectedModel ? "text-white" : "text-neutral-400"}`}
              >
                {currentDisplayName}
              </span>
              {selectedModel && (
                <span className="text-[10px] px-1.5 py-0.2 rounded bg-white/10 text-neutral-300 shrink-0">
                  {selectedModel.endsWith(".onnx") ? "ONNX" : "PTH"}
                </span>
              )}
            </div>
            {selectedModel && (
              <p className="text-[11px] text-neutral-400 truncate m-0 leading-tight mt-0.5">
                {modelFolder(selectedModel)} {hasIndexMatch(selectedModel) ? `• ${t("Index paired")}` : ""}
              </p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-1 shrink-0 text-neutral-400">
          <ChevronDown
            size={16}
            className={`transition-transform duration-200 ${open ? "rotate-180 text-white" : ""}`}
          />
        </div>
      </button>

      {menu}
    </div>
  );
}
