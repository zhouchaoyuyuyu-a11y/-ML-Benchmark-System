import { Stat, Tag } from "@/components/ui";
import { db } from "@/lib/store";
import AppConfigClient from "./AppConfigClient";

const checklist: { item: string; detail: string }[] = [
  {
    item: "Store listing assets",
    detail: "Icons, screenshots, feature graphics, and listing copy live in /app-store-assets — versioned with the codebase so every submission is reproducible.",
  },
  {
    item: "Native wrapper configuration",
    detail: "The iOS / Android wrapper project (bundle ids, splash, deep links) lives in app-wrapper/ and reads the download URLs configured on this page.",
  },
  {
    item: "Packaging & submission guide",
    detail: "Step-by-step build, signing, and store submission instructions are documented in docs/APP_PACKAGING.md for the operations team.",
  },
  {
    item: "PWA distribution",
    detail: "The web app ships its own manifest and service worker; the install prompt and download banner are controlled by the toggles on this page and are live on every deployment.",
  },
];

export default async function AdminAppPage() {
  const data = db();
  const config = data.app_config;
  const storeLinks = [config.ios_download_url, config.android_download_url, config.apk_download_url].filter(
    (u) => (u ?? "").trim().length > 0,
  ).length;

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">App & Downloads</h1>
      <p className="mt-1 text-sm text-mist">
        Distribution configuration for the ZOTAIX app — store links, version policy, and install surfaces. The
        public /download page reads this record; while a store link is empty, that channel presents the PWA
        install path as the designed route.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Latest version" value={config.latest_version} hint="Shown on /download and in-app" />
        <Stat
          label="Force update below"
          value={config.force_update_version || "—"}
          hint="Older clients are asked to update"
        />
        <Stat label="Store links set" value={`${storeLinks}/3`} hint="iOS · Android · APK" />
        <Stat label="Changelog entries" value={String(config.changelog.length)} hint="Published release notes" />
      </div>

      <div className="mt-8">
        <AppConfigClient config={config} />
      </div>

      <section className="mt-10">
        <h2 className="font-display text-lg text-porcelain">Changelog</h2>
        <p className="mt-1 text-xs text-mist">
          Release notes shipped with each version — rendered read-only here; new entries land with each release
          through the deployment pipeline.
        </p>
        <div className="mt-3 space-y-3">
          {config.changelog.map((entry) => (
            <div key={entry.version} className="zx-card p-4">
              <div className="flex flex-wrap items-center gap-3">
                <Tag tone={entry.version === config.latest_version ? "jade" : "default"}>v{entry.version}</Tag>
                <span className="text-xs text-mist">{entry.date}</span>
                {entry.version === config.latest_version && (
                  <span className="text-xs font-medium text-jade">current</span>
                )}
              </div>
              <ul className="mt-3 space-y-1.5">
                {entry.notes.map((note) => (
                  <li key={note} className="flex gap-2 text-sm text-mist">
                    <span className="text-gold">·</span>
                    <span>{note}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-10 pb-6">
        <h2 className="font-display text-lg text-porcelain">Distribution checklist</h2>
        <p className="mt-1 text-xs text-mist">Where every distribution artifact lives in the repository.</p>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          {checklist.map((c) => (
            <div key={c.item} className="zx-card p-4">
              <p className="text-sm text-porcelain">{c.item}</p>
              <p className="mt-1.5 text-xs leading-relaxed text-mist">{c.detail}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
