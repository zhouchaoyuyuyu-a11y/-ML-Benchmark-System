"use client";

import { useEffect, useState } from "react";

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
}

/** Registers the service worker and offers a lightweight PWA install prompt. */
export default function PWARegister({ enabled = true }: { enabled?: boolean }) {
  const [installEvent, setInstallEvent] = useState<BeforeInstallPromptEvent | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(() => {
        /* registration is best-effort */
      });
    }
    const handler = (e: Event) => {
      e.preventDefault();
      setInstallEvent(e as BeforeInstallPromptEvent);
    };
    window.addEventListener("beforeinstallprompt", handler);
    return () => window.removeEventListener("beforeinstallprompt", handler);
  }, []);

  if (!enabled || !installEvent || dismissed || (typeof window !== "undefined" && sessionStorage.getItem("zx_pwa_dismissed"))) {
    return null;
  }

  return (
    <div className="fixed inset-x-4 bottom-4 z-40 sm:left-auto sm:right-6 sm:w-96">
      <div className="zx-card flex items-center gap-3 p-4 shadow-2xl">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-gold/40 font-display text-gold">
          Z
        </div>
        <div className="flex-1">
          <p className="text-sm font-medium text-porcelain">Install ZOTAIX</p>
          <p className="text-xs text-mist">Add the app to your home screen for full-screen access.</p>
        </div>
        <button
          onClick={async () => {
            await installEvent.prompt();
            setInstallEvent(null);
          }}
          className="rounded-md bg-gold px-3 py-1.5 text-xs font-medium text-ink"
        >
          Install
        </button>
        <button
          onClick={() => {
            sessionStorage.setItem("zx_pwa_dismissed", "1");
            setDismissed(true);
          }}
          className="text-mist hover:text-porcelain"
          aria-label="Dismiss"
        >
          ✕
        </button>
      </div>
    </div>
  );
}
