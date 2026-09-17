"use client";

import { Check, ChevronDown, Mic2, RefreshCw, Search, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
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
  const dropdownRef = useRef<HTMLDivElement | null>(null);
  const searchInputRef = useRef<HTMLInputElement | null>(null);

  // Close on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    if (open) {
      document.addEventListener("mousedown", handleClickOutside);
      // Auto-focus search input when opening
      setTimeout(() => searchInputRef.current?.focus(), 50);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [open]);

  // Close on escape key
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape" && open) {
        setOpen(false);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [open]);

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

  return (
    <div ref={dropdownRef} className="relative w-full">
      {/* Dropdown Trigger Button */}
      <button
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
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-white/10 text-neutral-300 shrink-0">
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

      {/* Popover Menu */}
      {open && (
        <div
          role="listbox"
          className="absolute top-full left-0 right-0 mt-1.5 z-50 rounded-xl bg-[#171717] border border-white/15 shadow-2xl backdrop-blur-xl overflow-hidden animate-in fade-in slide-in-from-top-2 duration-150"
        >
          {/* Search Header */}
          <div className="p-2 border-b border-white/10 flex items-center gap-2 bg-black/40">
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
          <div className="max-h-60 overflow-y-auto p-1.5 space-y-1">
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
                          className={`text-[9px] font-mono px-1 py-0.2 rounded ${
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
          <div className="p-2 border-t border-white/10 bg-black/40 flex items-center justify-between gap-2 text-xs">
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
        </div>
      )}
    </div>
  );
}
