"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { NAV_SECTIONS } from "./nav";

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex flex-col w-64 shrink-0 bg-[#141414]/90 backdrop-blur-md border border-white/10 text-neutral-200 p-3 m-4 mr-0 rounded-2xl select-none">
      {/* Brand Header */}
      <div className="px-3 pt-2 pb-3 mb-1 border-b border-white/5 flex items-center justify-between">
        <Link href="/" prefetch={true} className="flex items-center gap-2.5 group">
          <span className="text-lg font-semibold tracking-tight text-white group-hover:text-neutral-200 transition-colors">
            Applio
          </span>
          <span className="text-[10px] uppercase font-medium tracking-wider px-1.5 py-0.5 rounded bg-white/10 text-neutral-400 border border-white/5">
            v3.6
          </span>
        </Link>
      </div>

      {/* Grouped Navigation */}
      <nav className="flex-1 overflow-y-auto space-y-4 pr-1 scrollbar-thin">
        {NAV_SECTIONS.map((section, sIdx) => (
          <div key={section.title || `sec-${sIdx}`} className="space-y-1">
            {section.title && (
              <div className="px-3 pt-1 pb-1 text-[10px] font-bold uppercase tracking-wider text-neutral-400">
                {section.title}
              </div>
            )}
            <ul className="space-y-0.5">
              {section.items.map((item) => {
                const Icon = item.icon;
                const active = pathname === item.to;
                return (
                  <li key={item.to}>
                    <Link
                      href={item.to}
                      prefetch={true}
                      className={`flex items-center gap-3 px-3 py-2 rounded-xl text-sm transition-all duration-150 relative ${
                        active
                          ? "bg-white/15 text-white font-medium shadow-xs"
                          : "text-neutral-400 hover:text-white hover:bg-white/5"
                      }`}
                    >
                      {active && <span className="absolute left-1 w-1 h-3.5 bg-white rounded-full" />}
                      <Icon
                        className={`w-4 h-4 shrink-0 transition-colors ${
                          active ? "text-white" : "text-neutral-400"
                        }`}
                      />
                      <span className="truncate">{item.label}</span>
                      {item.badge && (
                        <span className="ml-auto text-[10px] px-1.5 py-0.2 rounded bg-white/10 text-neutral-300">
                          {item.badge}
                        </span>
                      )}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>
    </aside>
  );
}
