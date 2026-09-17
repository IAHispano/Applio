"use client";

import type React from "react";
import { useEffect, useId, useState } from "react";

interface SliderFieldProps {
  id?: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  defaultValueText?: string;
  formatValue?: (val: number) => string;
  onChange: (val: number) => void;
  description?: string;
  disabled?: boolean;
  className?: string;
}

export default function SliderField({
  id: customId,
  label,
  value,
  min,
  max,
  step = 1,
  unit = "",
  defaultValueText,
  formatValue,
  onChange,
  description,
  disabled = false,
  className = "",
}: SliderFieldProps) {
  const generatedId = useId();
  const id = customId || `slider-${generatedId}`;
  const descId = `${id}-desc`;

  const [inputText, setInputText] = useState<string>(String(value));
  const [isFocused, setIsFocused] = useState(false);

  useEffect(() => {
    if (!isFocused) {
      setInputText(String(value));
    }
  }, [value, isFocused]);

  // Compute fill percentage for track styling
  const clampedValue = Math.min(Math.max(value, min), max);
  const percentage = max > min ? Math.max(0, Math.min(100, ((clampedValue - min) / (max - min)) * 100)) : 0;

  const ariaValueText = formatValue ? formatValue(value) : `${value}${unit ? ` ${unit}` : ""}`;

  const commitValue = (valStr: string) => {
    const parsed = Number.parseFloat(valStr);
    if (Number.isNaN(parsed)) {
      setInputText(String(value));
      return;
    }
    const clamped = Math.min(max, Math.max(min, parsed));
    const precision = step < 1 ? (step.toString().split(".")[1]?.length ?? 2) : 0;
    const rounded = Number(clamped.toFixed(precision));
    onChange(rounded);
    setInputText(String(rounded));
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const text = e.target.value;
    setInputText(text);
    const parsed = Number.parseFloat(text);
    if (!Number.isNaN(parsed) && parsed >= min && parsed <= max) {
      onChange(parsed);
    }
  };

  const handleInputBlur = () => {
    setIsFocused(false);
    commitValue(inputText);
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      commitValue(inputText);
      (e.target as HTMLInputElement).blur();
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      const next = Math.min(max, Math.round((value + step) * 1e4) / 1e4);
      onChange(next);
      setInputText(String(next));
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      const next = Math.max(min, Math.round((value - step) * 1e4) / 1e4);
      onChange(next);
      setInputText(String(next));
    }
  };

  const handleDecrement = (e: React.MouseEvent) => {
    e.preventDefault();
    const next = Math.max(min, Math.round((value - step) * 1e4) / 1e4);
    onChange(next);
    setInputText(String(next));
  };

  const handleIncrement = (e: React.MouseEvent) => {
    e.preventDefault();
    const next = Math.min(max, Math.round((value + step) * 1e4) / 1e4);
    onChange(next);
    setInputText(String(next));
  };

  return (
    <div className={`space-y-1.5 ${className}`}>
      <div className="flex items-center justify-between gap-2">
        <label
          htmlFor={id}
          className="text-xs font-medium text-neutral-300 m-0 select-none cursor-pointer flex-1 truncate"
        >
          {label}
        </label>

        {/* Manual numeric input box + unit */}
        <div className="flex items-center gap-1.5 shrink-0">
          <div className="relative flex items-center">
            <input
              type="text"
              inputMode="decimal"
              aria-label={`${label} value`}
              value={inputText}
              disabled={disabled}
              onFocus={() => setIsFocused(true)}
              onChange={handleInputChange}
              onBlur={handleInputBlur}
              onKeyDown={handleInputKeyDown}
              className="w-16 px-1.5 py-0.5 text-xs font-semibold font-mono text-center rounded-md bg-white/10 text-neutral-100 border border-white/15 hover:border-white/30 focus:border-white focus:bg-white/20 focus:outline-none transition-all tabular-nums disabled:opacity-40"
            />
          </div>
          {unit && <span className="text-[11px] text-neutral-400 select-none font-medium">{unit}</span>}
          {defaultValueText && (
            <span className="text-[10px] text-neutral-500 select-none hidden min-[480px]:inline">
              ({defaultValueText})
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-2">
        {/* Decrement fine-tune button */}
        <button
          type="button"
          onClick={handleDecrement}
          disabled={disabled || value <= min}
          aria-label={`Decrease ${label}`}
          className="slider-step-btn"
          style={{
            width: 28,
            height: 28,
            minWidth: 28,
            minHeight: 28,
            maxWidth: 28,
            maxHeight: 28,
            padding: 0,
            margin: 0,
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            borderRadius: 8,
            background: "rgba(255, 255, 255, 0.08)",
            border: "1px solid rgba(255, 255, 255, 0.15)",
            color: "#ffffff",
            cursor: disabled || value <= min ? "not-allowed" : "pointer",
            opacity: disabled || value <= min ? 0.35 : 1,
            transform: "none",
          }}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#ffffff"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
            className="shrink-0"
            style={{
              width: 14,
              height: 14,
              minWidth: 14,
              minHeight: 14,
              stroke: "#ffffff",
              display: "block",
            }}
          >
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
        </button>

        {/* Range Slider */}
        <div className="relative flex-1 flex items-center h-7">
          <input
            type="range"
            id={id}
            min={min}
            max={max}
            step={step}
            value={value}
            disabled={disabled}
            onChange={(e) => {
              const v = Number(e.target.value);
              onChange(v);
              setInputText(String(v));
            }}
            aria-label={label}
            aria-valuemin={min}
            aria-valuemax={max}
            aria-valuenow={value}
            aria-valuetext={ariaValueText}
            aria-describedby={description ? descId : undefined}
            style={{
              background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${percentage}%, var(--slider-track) ${percentage}%, var(--slider-track) 100%)`,
            }}
            className="w-full h-1.5 rounded-full appearance-none cursor-pointer transition-all focus-visible:outline-2 focus-visible:outline-white focus-visible:outline-offset-2"
          />
        </div>

        {/* Increment fine-tune button */}
        <button
          type="button"
          onClick={handleIncrement}
          disabled={disabled || value >= max}
          aria-label={`Increase ${label}`}
          className="slider-step-btn"
          style={{
            width: 28,
            height: 28,
            minWidth: 28,
            minHeight: 28,
            maxWidth: 28,
            maxHeight: 28,
            padding: 0,
            margin: 0,
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            borderRadius: 8,
            background: "rgba(255, 255, 255, 0.08)",
            border: "1px solid rgba(255, 255, 255, 0.15)",
            color: "#ffffff",
            cursor: disabled || value >= max ? "not-allowed" : "pointer",
            opacity: disabled || value >= max ? 0.35 : 1,
            transform: "none",
          }}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#ffffff"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
            className="shrink-0"
            style={{
              width: 14,
              height: 14,
              minWidth: 14,
              minHeight: 14,
              stroke: "#ffffff",
              display: "block",
            }}
          >
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
        </button>
      </div>

      {description && (
        <p id={descId} className="text-[11px] text-neutral-400 m-0 leading-tight">
          {description}
        </p>
      )}
    </div>
  );
}
