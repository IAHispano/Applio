"use client";

import type React from "react";

interface PageHeaderProps {
  title: string;
  description?: string;
  children?: React.ReactNode;
  className?: string;
}

export default function PageHeader({ title, description, children, className = "" }: PageHeaderProps) {
  return (
    <div
      className={`flex items-start justify-between gap-4 mb-6 pb-4 border-b border-white/10 flex-wrap ${className}`}
    >
      <div className="space-y-1">
        <h1 className="text-2xl font-bold tracking-tight text-white m-0">{title}</h1>
        {description && (
          <p className="text-sm text-neutral-400 m-0 max-w-2xl leading-relaxed">{description}</p>
        )}
      </div>
      {children && <div className="flex items-center gap-2 flex-wrap">{children}</div>}
    </div>
  );
}
