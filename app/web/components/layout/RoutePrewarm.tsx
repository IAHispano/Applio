"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { MENU } from "./nav";

// Next disables viewport prefetching in development (it would compile every
// target page), so the first visit to a tab pays for its compile + chunk and
// feels frozen. Warm the routes ourselves once the current page is idle, one at
// a time, so the queue never competes with whatever the user is doing now.
export default function RoutePrewarm() {
  const router = useRouter();

  useEffect(() => {
    const routes = MENU.map((entry) => entry.to);
    let index = 0;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    const warmNext = (): void => {
      if (cancelled || index >= routes.length) return;
      const target = routes[index++];
      try {
        router.prefetch(target);
      } catch {
        /* a failed warm-up is never worth surfacing */
      }
      timer = setTimeout(warmNext, 250);
    };

    timer = setTimeout(warmNext, 1000);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [router]);

  return null;
}
