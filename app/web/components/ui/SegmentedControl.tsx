"use client";

import type { LucideIcon } from "lucide-react";

interface Option<T extends string> {
  value: T;
  label: string;
  icon?: LucideIcon;
}

interface SegmentedControlProps<T extends string> {
  value: T;
  options: Option<T>[];
  onChange: (value: T) => void;
  ariaLabel?: string;
  /** Wires each tab to its panel: id="tab-<value>", aria-controls="panel-<value>". */
  tabPanels?: boolean;
}

export default function SegmentedControl<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
  tabPanels = false,
}: SegmentedControlProps<T>) {
  return (
    <div className="segmented" role="tablist" aria-label={ariaLabel}>
      {options.map((option) => {
        const Icon = option.icon;
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="tab"
            id={tabPanels ? `tab-${option.value}` : undefined}
            aria-controls={tabPanels ? `panel-${option.value}` : undefined}
            aria-selected={active}
            className={`segmented-item ${active ? "is-active" : ""}`}
            onClick={() => onChange(option.value)}
          >
            {Icon && <Icon size={14} />}
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
