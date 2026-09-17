"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function ToolsPage() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/extra");
  }, [router]);

  return null;
}
