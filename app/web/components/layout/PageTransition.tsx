"use client";

import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

export default function PageTransition({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  // Keying on the path remounts the wrapper, which replays the enter
  // animation on every tab switch.
  return (
    <div key={pathname} className="page-transition h-full">
      {children}
    </div>
  );
}
