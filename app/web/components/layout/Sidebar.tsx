"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { MENU } from "./nav";

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="flex flex-col w-64 bg-[#1c1c1c]/10 border border-white/10 text-gray-100 p-4 m-4 mr-0 rounded-xl">
      <Link href="/" className="px-2.5 pb-4">
        <span className="title text-2xl font-bold tracking-tight">Applio</span>
      </Link>
      <nav className="flex-1 overflow-auto">
        <ul className="space-y-2">
          {MENU.filter((item) => item.to !== "/").map((item) => {
            const Icon = item.icon;
            const active = pathname === item.to;
            return (
              <li key={item.to}>
                <Link
                  href={item.to}
                  className={`flex items-center justify-start space-x-3 p-2.5 rounded-lg ${
                    active ? "bg-white/10" : ""
                  } hover:bg-white/10 transition-colors duration-200 opacity-70`}
                >
                  <Icon className="w-5 h-5 shrink-0" />
                  <span>{item.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </div>
  );
}
