import type { Metadata } from "next";
import type { ReactNode } from "react";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "App Privacy Notice",
  description:
    "Privacy notice for the ZOTAIX app and installable web app: persistent login, device permissions, push notification behavior, store distribution, and update policy.",
  path: "/legal/app",
});

function Sec({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="font-display text-lg text-porcelain">
        {n}. {title}
      </h2>
      <div className="space-y-3 text-sm leading-relaxed text-mist">{children}</div>
    </section>
  );
}

export default async function AppPrivacyPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const doc = data.legal_docs.find((d) => d.slug === "app");
  const app = data.app_config;

  return (
    <article className="space-y-10">
      <header>
        <h1 className="font-display text-3xl text-porcelain sm:text-4xl">
          {zh ? "App 隐私声明" : "App Privacy Notice"}
        </h1>
        <p className="mt-3 text-xs uppercase tracking-[0.2em] text-gold">
          {zh ? "版本" : "Version"} {doc?.version ?? "1.0"} · {zh ? "生效日期" : "Effective date"}{" "}
          {doc?.effective_date ?? "2026-05-01"}
        </p>
        <p className="mt-4 text-sm leading-relaxed text-mist">
          {zh
            ? "本声明补充隐私政策，说明 ZOTAIX 应用（可安装的 Web 应用及其商店封装版本）在你的设备上的数据行为。"
            : "This notice supplements the Privacy Policy for the ZOTAIX app — the installable web app and its store-wrapped editions — and describes what the app does on your device."}
        </p>
      </header>

      <Sec n={1} title={zh ? "适用范围" : "Scope"}>
        <p>
          The ZOTAIX app is the same platform you use in the browser, packaged for installation: a progressive web app
          you can add to your home screen, and wrapped editions distributed through app stores where the operator has
          published them. All editions talk to the same service, apply the same Privacy Policy, and add only the
          device-level behaviors described here. Nothing in the app collects data the website does not.
        </p>
      </Sec>

      <Sec n={2} title={zh ? "设备上的数据" : "Data on your device"}>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>
            <span className="text-porcelain">Persistent login.</span> The app keeps your session cookie so you stay
            signed in between launches, exactly as a browser would. Signing out clears it. On a shared device, sign out
            before handing the device over.
          </li>
          <li>
            <span className="text-porcelain">Offline shell.</span> The app caches its interface files on your device so
            it opens quickly and shows a designed offline page without a connection. The cache contains interface
            assets, not your personal content.
          </li>
          <li>
            <span className="text-porcelain">Preferences.</span> Language, age-gate confirmation, and install prompt
            dismissal are stored on-device as described in the Cookie Policy.
          </li>
        </ul>
      </Sec>

      <Sec n={3} title={zh ? "推送通知" : "Push notifications"}>
        <p>
          Push notifications operate only where two things are true: the operator has configured a push channel for
          your platform, and you have explicitly granted the notification permission. Where either is absent, the app
          uses in-app announcements instead — quota updates, co-creation progress, and concierge replies appear when
          you open the app, and nothing is sent to your device in the background. You can withdraw notification
          permission at any time in your device settings, and the app falls back to in-app announcements from that
          moment.
        </p>
      </Sec>

      <Sec n={4} title={zh ? "设备权限" : "Device permissions"}>
        <p>The app requests permissions only at the moment a feature needs them, and every feature has a path without the permission:</p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>
            <span className="text-porcelain">Camera</span> — only if you choose to scan a ZOTAIX QR code in-app; you
            can always type the serial instead.
          </li>
          <li>
            <span className="text-porcelain">Photos / storage</span> — only if you choose to save an emotional card or
            certificate image to your device; sharing by link needs no permission.
          </li>
          <li>
            <span className="text-porcelain">Notifications</span> — only if you enable push, as described above.
          </li>
        </ul>
        <p>
          The app requests no location, contacts, microphone, or background data access. A permission you decline is
          never re-prompted outside your explicit retry.
        </p>
      </Sec>

      <Sec n={5} title={zh ? "商店分发说明" : "Store distribution"}>
        <p>
          Store listings are configured by the operations team. Where a store edition is published, the store's own
          privacy label reflects this notice, and the operator's legal name, registered address, and contact are shown
          in the operator information section of the listing. Where no store edition is published for your platform,
          the installable web app from this site is the official app, and the same information is published in the
          operator information section of the WeChat official account. Only apps from these official sources are
          genuine; the operator distributes no builds through third-party download portals.
        </p>
      </Sec>

      <Sec n={6} title={zh ? "更新策略" : "Update policy"}>
        <p>
          The current app version is {app.latest_version}, with the changelog published on the download page. The
          installable web app updates itself when you open it — no action needed. Store editions update through the
          store's normal mechanism. Versions at or above {app.force_update_version ?? app.latest_version} remain fully
          supported; when a version falls below the supported baseline, the app asks you to update before signing in,
          because older versions cannot be guaranteed to apply current privacy and compliance behavior. Updates never
          silently expand permissions: a new permission is always requested in context, per Section 4.
        </p>
      </Sec>

      <Sec n={7} title={zh ? "联系" : "Contact"}>
        <p>
          Questions about the app's data behavior: concierge@zotaix.example, or the channels on the Contact page. For
          privacy rights (access, export, deletion), the Privacy Policy applies to app and website alike.
        </p>
      </Sec>
    </article>
  );
}
