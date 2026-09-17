"use client";

import { ChevronLeft, ChevronRight, Maximize, Minimize, Minus, RefreshCcw, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

interface WindowControls {
  minimize: () => void;
  toggleMaximize: () => void;
  close: () => void;
}

function useWindowControls(): WindowControls | null {
  const [controls, setControls] = useState<WindowControls | null>(null);
  useEffect(() => {
    const bridge = (window as unknown as { applio?: { controls?: WindowControls } }).applio;
    if (bridge?.controls) setControls(bridge.controls);
  }, []);
  return controls;
}

export default function TitleBar() {
  const router = useRouter();
  const controls = useWindowControls();
  const [maximized, setMaximized] = useState(false);

  async function toggleMaximize() {
    controls?.toggleMaximize();
    setMaximized((v) => !v);
  }

  function goBack() {
    if (window.history.length > 1) router.back();
    else router.push("/");
  }

  function goForward() {
    router.forward();
  }

  return (
    <div className="absolute top-0 right-0 select-none overflow-hidden p-2 pt-3 w-full [-webkit-app-region:drag]">
      <div className="flex justify-between items-center px-2">
        <div className="justify-start ml-auto w-full gap-2 flex px-2 [-webkit-app-region:no-drag]">
          <button
            type="button"
            className="bg-transparent! border-0! p-1!"
            onClick={() => window.location.reload()}
            aria-label="Reload"
          >
            <RefreshCcw className="text-neutral-300 slow duration-200 hover:text-white w-4 h-4" />
          </button>
          <button type="button" className="bg-transparent! border-0! p-1!" onClick={goBack} aria-label="Back">
            <ChevronLeft className="text-neutral-300 slow duration-200 hover:text-white w-5 h-5" />
          </button>
          <button
            type="button"
            className="bg-transparent! border-0! p-1!"
            onClick={goForward}
            aria-label="Forward"
          >
            <ChevronRight className="text-neutral-300 slow duration-200 hover:text-white w-5 h-5" />
          </button>
        </div>
        <p className="text-sm text-neutral-300 title font-medium flex mx-auto w-full">Applio App</p>
        {controls && (
          <div className="justify-end flex gap-3 [-webkit-app-region:no-drag]">
            <button
              type="button"
              className="bg-transparent! border-0! p-1!"
              onClick={() => controls.minimize()}
              aria-label="Minimize"
            >
              <Minus className="text-neutral-300 hover:text-neutral-200 slow w-5 h-5" />
            </button>
            <button
              type="button"
              className="bg-transparent! border-0! p-1!"
              onClick={toggleMaximize}
              aria-label="Maximize"
            >
              {maximized ? (
                <Minimize className="text-neutral-300 hover:text-neutral-200 slow w-3.5 h-3.5" />
              ) : (
                <Maximize className="text-neutral-300 hover:text-neutral-200 slow w-3.5 h-3.5" />
              )}
            </button>
            <button
              type="button"
              className="bg-transparent! border-0! p-1!"
              onClick={() => controls.close()}
              aria-label="Close"
            >
              <X className="text-neutral-300 hover:text-red-400 slow w-5 h-5" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
