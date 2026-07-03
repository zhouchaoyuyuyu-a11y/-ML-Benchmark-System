import type { Metadata } from "next";
import type { ReactNode } from "react";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Cookie Policy",
  description:
    "The cookies and browser storage ZOTAIX uses: session, visitor quota, and locale cookies, plus local storage for age confirmation and install prompts. No third-party advertising cookies.",
  path: "/legal/cookies",
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

const rows: { name: string; kind: string; purpose: string; lifetime: string }[] = [
  {
    name: "zx_session",
    kind: "Cookie · essential",
    purpose:
      "Keeps you signed in to your account so your drafts, profile, quotas, and Reserve records belong to you. Set only after you sign in.",
    lifetime: "Session-based; cleared on sign-out.",
  },
  {
    name: "zx_visitor",
    kind: "Cookie · essential",
    purpose:
      "An anonymous visitor identifier that enforces the guest daily quota for the AI concierge. It contains no personal data and is not linked to an identity.",
    lifetime: "Persists until cleared, so the guest quota works across visits.",
  },
  {
    name: "zx_lang",
    kind: "Cookie · preference",
    purpose: "Remembers your language choice (English or 中文) so every page renders in your preferred locale.",
    lifetime: "Persists until cleared or changed.",
  },
  {
    name: "zx_age_confirmed",
    kind: "localStorage · compliance",
    purpose:
      "Records that you confirmed you are of legal drinking age when entering alcohol-related areas, so the age gate does not reappear on every page.",
    lifetime: "Stays in your browser until you clear site data.",
  },
  {
    name: "zx_pwa_dismissed",
    kind: "sessionStorage · preference",
    purpose:
      "Remembers that you dismissed the web-app install prompt so it does not reappear during the same browsing session.",
    lifetime: "Cleared automatically when the browser session ends.",
  },
];

export default async function CookiesPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const doc = db().legal_docs.find((d) => d.slug === "cookies");

  return (
    <article className="space-y-10">
      <header>
        <h1 className="font-display text-3xl text-porcelain sm:text-4xl">{zh ? "Cookie 政策" : "Cookie Policy"}</h1>
        <p className="mt-3 text-xs uppercase tracking-[0.2em] text-gold">
          {zh ? "版本" : "Version"} {doc?.version ?? "1.0"} · {zh ? "生效日期" : "Effective date"}{" "}
          {doc?.effective_date ?? "2026-05-01"}
        </p>
        <p className="mt-4 text-sm leading-relaxed text-mist">
          {zh
            ? "ZOTAIX 只使用运行平台所必需的少量 Cookie 与浏览器存储。没有广告 Cookie，没有第三方追踪。"
            : "ZOTAIX uses a deliberately small set of cookies and browser storage — only what the platform needs to run. No advertising cookies, no third-party tracking."}
        </p>
      </header>

      <Sec n={1} title={zh ? "什么是 Cookie 与浏览器存储" : "What cookies and browser storage are"}>
        <p>
          Cookies are small text entries a website asks your browser to keep and send back on later requests, so the
          site can recognize your session or your preferences. Browser storage (localStorage and sessionStorage) works
          similarly but stays on your device and is never transmitted automatically. ZOTAIX uses both, sparingly, for
          the exact purposes listed below and for nothing else.
        </p>
      </Sec>

      <Sec n={2} title={zh ? "我们设置的项目" : "Everything we set"}>
        <p>
          This is the complete list. If an entry is not in this table, ZOTAIX did not set it.
        </p>
        <div className="overflow-x-auto rounded-lg border border-hairline">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead>
              <tr className="border-b border-hairline bg-veil">
                <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-porcelain">
                  {zh ? "名称" : "Name"}
                </th>
                <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-porcelain">
                  {zh ? "类型" : "Type"}
                </th>
                <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-porcelain">
                  {zh ? "用途" : "Purpose"}
                </th>
                <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-porcelain">
                  {zh ? "有效期" : "Lifetime"}
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name} className="border-b border-hairline last:border-0 align-top">
                  <td className="px-4 py-3 font-mono text-xs text-gold">{r.name}</td>
                  <td className="px-4 py-3 text-xs text-porcelain">{r.kind}</td>
                  <td className="px-4 py-3 text-xs leading-relaxed text-mist">{r.purpose}</td>
                  <td className="px-4 py-3 text-xs leading-relaxed text-mist">{r.lifetime}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Sec>

      <Sec n={3} title={zh ? "为什么这些是必要的" : "Why these are necessary"}>
        <p>
          <span className="text-porcelain">zx_session</span> is what makes an account an account: without it, the
          platform could not keep your drafts, memory settings, quotas, and Reserve records attached to you between
          pages. <span className="text-porcelain">zx_visitor</span> exists so guests can try the concierge fairly —
          the daily guest quota has to be counted somewhere, and an anonymous identifier is the least invasive way to
          do it. <span className="text-porcelain">zx_lang</span> saves your language choice so you are not asked on
          every visit. <span className="text-porcelain">zx_age_confirmed</span> supports the alcohol age gate required
          by the Alcohol Compliance Notice, and <span className="text-porcelain">zx_pwa_dismissed</span> simply keeps
          the install prompt from repeating after you have closed it.
        </p>
      </Sec>

      <Sec n={4} title={zh ? "没有第三方广告 Cookie" : "No third-party advertising cookies"}>
        <p>
          ZOTAIX sets no advertising cookies, embeds no ad networks, and includes no cross-site tracking pixels. Links
          to our official social accounts are plain links: no social platform script runs on ZOTAIX pages, and no
          social platform can read your ZOTAIX activity through this site. Where the operator has configured external
          processors (AI providers, payment processors), those integrations run server-side and do not set cookies in
          your browser through ZOTAIX pages.
        </p>
      </Sec>

      <Sec n={5} title={zh ? "如何管理与清除" : "How to manage and clear"}>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>
            <span className="text-porcelain">Sign out</span> to end your session and invalidate zx_session immediately.
          </li>
          <li>
            <span className="text-porcelain">Clear site data</span> in your browser settings (usually under Privacy →
            Cookies and site data) to remove all entries in the table above at once, including localStorage.
          </li>
          <li>
            <span className="text-porcelain">Block cookies</span> for this site if you prefer: the public pages remain
            readable, but signing in, guest quota tracking, language memory, and the age gate confirmation will not
            work, since they depend on the entries above.
          </li>
        </ul>
        <p>
          Clearing zx_age_confirmed means the age gate will ask again on your next visit to an alcohol-related area —
          that is the intended behavior, not an error.
        </p>
      </Sec>

      <Sec n={6} title={zh ? "变更与联系" : "Changes and contact"}>
        <p>
          If the set of cookies ever changes, this table is updated and the version number above is raised before the
          change takes effect. Questions about cookies and storage: concierge@zotaix.example, or the channels on the
          Contact page.
        </p>
      </Sec>
    </article>
  );
}
