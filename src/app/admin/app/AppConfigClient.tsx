"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui";
import type { AppConfig } from "@/lib/types";

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

export default function AppConfigClient({ config }: { config: AppConfig }) {
  const router = useRouter();
  const [iosUrl, setIosUrl] = useState(config.ios_download_url ?? "");
  const [androidUrl, setAndroidUrl] = useState(config.android_download_url ?? "");
  const [apkUrl, setApkUrl] = useState(config.apk_download_url ?? "");
  const [latestVersion, setLatestVersion] = useState(config.latest_version);
  const [forceVersion, setForceVersion] = useState(config.force_update_version ?? "");
  const [pwaEnabled, setPwaEnabled] = useState(config.pwa_enabled);
  const [showBanner, setShowBanner] = useState(config.show_download_banner);
  const [installPrompt, setInstallPrompt] = useState(config.install_prompt_enabled);
  const [downloadsEnabled, setDownloadsEnabled] = useState(config.downloads_enabled);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function touch() {
    setSaved(false);
    setError(null);
  }

  const toggles: { label: string; hint: string; value: boolean; set: (v: boolean) => void }[] = [
    { label: "PWA enabled", hint: "Service worker registration and installable manifest", value: pwaEnabled, set: setPwaEnabled },
    { label: "Show download banner", hint: "Sitewide banner pointing to /download", value: showBanner, set: setShowBanner },
    { label: "Install prompt enabled", hint: "Browser install prompt on eligible devices", value: installPrompt, set: setInstallPrompt },
    { label: "Downloads enabled", hint: "Master switch for the /download page entries", value: downloadsEnabled, set: setDownloadsEnabled },
  ];

  async function save() {
    setSaving(true);
    setSaved(false);
    setError(null);
    try {
      const res = await fetch("/api/admin/config", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          section: "app",
          patch: {
            ios_download_url: iosUrl,
            android_download_url: androidUrl,
            apk_download_url: apkUrl,
            latest_version: latestVersion,
            force_update_version: forceVersion,
            pwa_enabled: pwaEnabled,
            show_download_banner: showBanner,
            install_prompt_enabled: installPrompt,
            downloads_enabled: downloadsEnabled,
          },
        }),
      });
      const data = await res.json().catch(() => ({ ok: false }));
      if (!data.ok) {
        setError(data.error ?? "Save failed — check the values and try again.");
      } else {
        setSaved(true);
        router.refresh();
      }
    } catch {
      setError("Network error — the configuration was not saved.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="zx-card p-5">
        <h2 className="font-display text-lg text-porcelain">Distribution URLs & versions</h2>
        <p className="mt-1 text-xs text-mist">
          A channel with an empty URL presents the PWA install path on /download instead of a store button.
        </p>
        <div className="mt-4 space-y-4">
          <label className="block">
            <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">iOS App Store URL</span>
            <input
              value={iosUrl}
              onChange={(e) => { touch(); setIosUrl(e.target.value); }}
              className={inputCls}
              placeholder="https://apps.apple.com/…"
            />
          </label>
          <label className="block">
            <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Android Play Store URL</span>
            <input
              value={androidUrl}
              onChange={(e) => { touch(); setAndroidUrl(e.target.value); }}
              className={inputCls}
              placeholder="https://play.google.com/store/apps/…"
            />
          </label>
          <label className="block">
            <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Direct APK URL</span>
            <input
              value={apkUrl}
              onChange={(e) => { touch(); setApkUrl(e.target.value); }}
              className={inputCls}
              placeholder="https://… (signed APK for direct distribution)"
            />
          </label>
          <div className="grid gap-4 sm:grid-cols-2">
            <label className="block">
              <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Latest version</span>
              <input
                value={latestVersion}
                onChange={(e) => { touch(); setLatestVersion(e.target.value); }}
                className={`${inputCls} font-mono`}
                placeholder="1.4.0"
              />
            </label>
            <label className="block">
              <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Force update below</span>
              <input
                value={forceVersion}
                onChange={(e) => { touch(); setForceVersion(e.target.value); }}
                className={`${inputCls} font-mono`}
                placeholder="1.2.0"
              />
            </label>
          </div>
        </div>
      </div>

      <div className="zx-card flex flex-col p-5">
        <h2 className="font-display text-lg text-porcelain">Install surfaces</h2>
        <p className="mt-1 text-xs text-mist">Switches for every place the app offers itself to a visitor.</p>
        <div className="mt-4 space-y-3">
          {toggles.map((t) => (
            <label
              key={t.label}
              className="flex cursor-pointer items-start gap-3 rounded-md border border-hairline px-3 py-3 transition-colors hover:border-gold/40"
            >
              <input
                type="checkbox"
                checked={t.value}
                onChange={(e) => { touch(); t.set(e.target.checked); }}
                className="mt-0.5 h-4 w-4 rounded border-hairline bg-ink accent-gold"
              />
              <span>
                <span className="block text-sm text-porcelain">{t.label}</span>
                <span className="mt-0.5 block text-xs text-mist">{t.hint}</span>
              </span>
            </label>
          ))}
        </div>
        <div className="mt-auto flex flex-wrap items-center gap-3 pt-5">
          <Button onClick={save} disabled={saving}>
            {saving ? "Saving…" : "Save app configuration"}
          </Button>
          {saved && <span className="text-sm font-medium text-jade">Saved ✓</span>}
          {error && <span className="text-sm text-ember">{error}</span>}
        </div>
      </div>
    </div>
  );
}
