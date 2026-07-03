import Link from "next/link";
import type { Metadata } from "next";
import { ButtonLink, Card, Meridian, Notice, PageHero, Section, SectionHeader, Stat, Tag } from "@/components/ui";
import QRCodeBox from "@/components/QRCodeBox";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";
import type { SocialAccount } from "@/lib/types";
import DownloadClient from "./DownloadClient";

export const metadata: Metadata = pageMetadata({
  title: "Download the ZOTAIX app",
  description:
    "Install ZOTAIX on iPhone, Android, or desktop: official store channels, direct APK, and a one-tap web app install. Same AI concierge, Reserve archive, and co-creation pool on every device.",
  path: "/download",
  keywords: ["ZOTAIX app", "download", "PWA", "iOS", "Android", "APK"],
});

function socialHref(account: SocialAccount): string {
  if (!account.tracking_params) return account.official_url;
  return `${account.official_url}${account.official_url.includes("?") ? "&" : "?"}${account.tracking_params}`;
}

const storeButtonCls =
  "inline-flex items-center justify-center gap-2 rounded-md border border-gold bg-gold px-5 py-2.5 text-sm font-medium text-ink transition-colors hover:bg-gold-deep hover:text-porcelain";

export default async function DownloadPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const app = data.app_config;
  const distributionNote = data.cms_blocks.find((b) => b.key === "download.beta.note" && b.enabled);
  const social = data.social_accounts.filter((s) => s.enabled).sort((a, b) => a.display_order - b.display_order);

  const channels: {
    key: string;
    icon: string;
    name: string;
    store: string;
    url?: string;
    cta: string;
    fallback: string;
  }[] = [
    {
      key: "ios",
      icon: "◉",
      name: "iOS",
      store: "App Store",
      url: app.ios_download_url,
      cta: zh ? "在 App Store 获取" : "Get it on the App Store",
      fallback: zh
        ? "App Store 分发链接由运营团队在管理后台配置，配置完成后下载按钮会自动出现在这里。iPhone 用户现在即可安装网页应用——同一账号、同一礼宾、同一档案。"
        : "The App Store distribution link is configured by the operations team in the admin console; once set, the download button appears here automatically. On iPhone, the web app installs today — same account, same concierge, same archive.",
    },
    {
      key: "android",
      icon: "▶",
      name: "Android",
      store: "Google Play",
      url: app.android_download_url,
      cta: zh ? "在 Google Play 获取" : "Get it on Google Play",
      fallback: zh
        ? "Google Play 分发链接由运营团队在管理后台配置，配置完成后下载按钮会自动出现。Android Chrome 现在即可一键安装网页应用，体验与商店版本一致。"
        : "The Google Play distribution link is configured by the operations team in the admin console; once set, the download button appears here. Android Chrome installs the web app in one tap today, matching the store build.",
    },
    {
      key: "apk",
      icon: "◇",
      name: "Android APK",
      store: zh ? "官方直装包" : "Direct download",
      url: app.apk_download_url,
      cta: zh ? "下载 APK 安装包" : "Download the APK",
      fallback: zh
        ? "官方 APK 直装包及其签名校验信息由运营团队在管理后台配置，配置后此处提供直接下载。无法访问应用商店的设备可先安装网页应用。"
        : "The signed official APK and its checksum are configured by the operations team in the admin console; once set, the direct download appears here. Devices without store access can install the web app now.",
    },
  ];

  const deepLinks = [
    { href: "/reserve", label: zh ? "档案馆 Reserve" : "Reserve archive", route: "zotaix://reserve" },
    { href: "/co-create", label: zh ? "共创铸造池" : "Co-creation pool", route: "zotaix://co-create" },
    { href: "/concierge", label: zh ? "AI 礼宾" : "AI concierge", route: "zotaix://concierge" },
  ];

  const installSteps = [
    {
      title: zh ? "选择渠道" : "Pick a channel",
      desc: zh
        ? "App Store、Google Play、官方 APK 或网页应用——四条渠道指向同一个 ZOTAIX。"
        : "App Store, Google Play, the official APK, or the web app — four channels, one ZOTAIX.",
    },
    {
      title: zh ? "安装并打开" : "Install and open",
      desc: zh
        ? "按本页指南安装；首次打开即可进入 AI 礼宾，游客也能直接对话。"
        : "Install with the guides on this page; on first open the AI concierge is ready — guests can talk right away.",
    },
    {
      title: zh ? "登录同步档案" : "Sign in to sync",
      desc: zh
        ? "登录后你的草稿、档案记录与会员权益在所有设备间同步。"
        : "Sign in and your drafts, Reserve records, and membership benefits sync across every device.",
    },
    {
      title: zh ? "从情绪开始创造" : "Create from an emotion",
      desc: zh
        ? "先创造并保存一个属于你的对象，再决定它是否成为实物。"
        : "Create and save a personalized object first — then decide whether it becomes physical.",
    },
  ];

  return (
    <>
      <PageHero
        eyebrow={zh ? "ZOTAIX App · 官方下载" : "ZOTAIX App · Official download"}
        title={zh ? "把 AI 礼宾装进口袋" : "Put the AI concierge in your pocket"}
        description={
          zh
            ? "iOS、Android、官方 APK 与网页应用四条安装渠道。同一个账号，同一套档案：AI 礼宾、共创池、档案馆与会员权益在每台设备上保持一致。"
            : "Four install channels — iOS, Android, official APK, and the web app. One account, one archive: the AI concierge, co-creation pool, Reserve records, and membership benefits stay consistent on every device."
        }
      >
        <ButtonLink href="#pwa" variant="gold">
          {zh ? "立即安装网页应用" : "Install the web app now"}
        </ButtonLink>
        <ButtonLink href="#channels" variant="outline">
          {zh ? "查看全部渠道" : "See all channels"}
        </ButtonLink>
        <ButtonLink href="/concierge" variant="outline">
          {zh ? "先在浏览器体验" : "Try it in the browser first"}
        </ButtonLink>
      </PageHero>

      {/* Version stats */}
      <Section className="py-10 sm:py-12">
        <div className="grid gap-4 sm:grid-cols-3">
          <Stat label={zh ? "最新版本" : "Latest version"} value={`v${app.latest_version}`} hint={app.changelog[0]?.date} />
          <Stat
            label={zh ? "网页应用" : "Web app"}
            value={app.pwa_enabled ? (zh ? "可安装" : "Installable") : zh ? "浏览器访问" : "Browser access"}
            hint={zh ? "离线壳 · 全屏运行" : "Offline shell · full-screen"}
          />
          <Stat
            label={zh ? "最低支持版本" : "Minimum supported"}
            value={app.force_update_version ? `v${app.force_update_version}` : `v${app.latest_version}`}
            hint={zh ? "更低版本会在登录前提示更新" : "Older builds are asked to update before sign-in"}
          />
        </div>
      </Section>

      {/* Store channels */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section id="channels" className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "安装渠道" : "Install channels"}
            title={zh ? "应用商店与官方直装" : "Store listings and direct install"}
            description={
              zh
                ? "商店分发由运营团队统一管理；每条渠道的按钮直接读取平台配置。"
                : "Store distribution is managed centrally by the operations team; each channel’s button reads live platform configuration."
            }
          />
          {distributionNote && (
            <div className="mt-6">
              <Notice tone="gold" title={zh ? "分发说明" : "Distribution note"}>
                {distributionNote.content}
              </Notice>
            </div>
          )}
          <div className="mt-8 grid gap-4 lg:grid-cols-3">
            {channels.map((c) => (
              <Card key={c.key} className="flex h-full flex-col">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <span className="flex h-10 w-10 items-center justify-center rounded-lg border border-hairline text-lg text-gold">
                      {c.icon}
                    </span>
                    <div>
                      <p className="font-display text-lg text-porcelain">{c.name}</p>
                      <p className="text-xs text-mist">{c.store}</p>
                    </div>
                  </div>
                  {c.url ? (
                    <Tag tone="jade">{zh ? "已上线" : "Live"}</Tag>
                  ) : (
                    <Tag>{zh ? "运营配置" : "Operations-configured"}</Tag>
                  )}
                </div>
                {c.url ? (
                  <>
                    <p className="mt-3 flex-1 text-sm leading-relaxed text-mist">
                      {zh
                        ? "官方渠道分发，自动更新，与网页应用共享账号与档案。"
                        : "Official channel distribution with automatic updates, sharing your account and archive with the web app."}
                    </p>
                    <div className="mt-4">
                      <a href={c.url} target="_blank" rel="noopener noreferrer" className={storeButtonCls}>
                        {c.cta}
                      </a>
                    </div>
                  </>
                ) : (
                  <>
                    <p className="mt-3 flex-1 text-sm leading-relaxed text-mist">{c.fallback}</p>
                    <div className="mt-4">
                      <ButtonLink href="#pwa" variant="outline">
                        {zh ? "先装网页应用" : "Install the web app"}
                      </ButtonLink>
                    </div>
                  </>
                )}
              </Card>
            ))}
          </div>
        </Section>
      </div>

      {/* PWA install */}
      <Section id="pwa" className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "网页应用" : "Web app"}
          title={zh ? "三十秒装好，无需商店" : "Installed in thirty seconds, no store required"}
          description={
            zh
              ? "ZOTAIX 网页应用支持添加到主屏幕、独立窗口运行与离线壳。以下指南覆盖 iOS Safari、Android Chrome 与桌面 Chrome / Edge。"
              : "The ZOTAIX web app adds to your home screen, runs in its own window, and keeps an offline shell. The guides below cover iOS Safari, Android Chrome, and desktop Chrome / Edge."
          }
        />
        <div className="mt-8">
          <DownloadClient zh={zh} />
        </div>
      </Section>

      {/* Installation steps + deep links */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "安装之后" : "After installing"}
            title={zh ? "四步进入你的定制链" : "Four steps into your customization chain"}
          />
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {installSteps.map((s, i) => (
              <Card key={s.title}>
                <p className="font-display text-3xl text-gold/60">{String(i + 1).padStart(2, "0")}</p>
                <p className="font-display mt-2 text-base text-porcelain">{s.title}</p>
                <p className="mt-2 text-sm leading-relaxed text-mist">{s.desc}</p>
              </Card>
            ))}
          </div>
          <div className="mt-8 grid gap-4 lg:grid-cols-2">
            <Card>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
                {zh ? "深度链接" : "Deep links"}
              </p>
              <p className="mt-2 text-sm leading-relaxed text-mist">
                {zh
                  ? "应用内路由与网页共用同一地址结构，分享出去的链接会直接打开对应页面："
                  : "App routes share the web’s address structure, so shared links open the matching screen directly:"}
              </p>
              <ul className="mt-3 space-y-2">
                {deepLinks.map((d) => (
                  <li key={d.href} className="flex flex-wrap items-center justify-between gap-2 border-b border-hairline pb-2 last:border-0 last:pb-0">
                    <Link href={d.href} className="text-sm text-porcelain transition-colors hover:text-gold">
                      {d.label} →
                    </Link>
                    <code className="rounded bg-veil px-2 py-0.5 text-xs text-mist">{d.route}</code>
                  </li>
                ))}
              </ul>
            </Card>
            <Card className="flex flex-col items-center justify-center gap-3 text-center">
              <QRCodeBox seed="https://zotaix-web.vercel.app/download" label="Scan to open" />
              <p className="max-w-xs text-xs leading-relaxed text-mist">
                {zh
                  ? "用手机相机扫码即可打开本页，在移动端完成安装。"
                  : "Scan with your phone camera to open this page and finish the install on mobile."}
              </p>
            </Card>
          </div>
        </Section>
      </div>

      {/* Changelog */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "版本记录" : "Changelog"}
          title={zh ? `当前版本 v${app.latest_version}` : `Now shipping v${app.latest_version}`}
          description={
            zh
              ? "每个版本聚焦一件事：让对象的创造、保存与铸造更顺手。"
              : "Every release focuses on one thing: making objects easier to create, keep, and cast."
          }
        />
        <div className="mt-8 grid gap-4 lg:grid-cols-3">
          {app.changelog.map((entry, i) => (
            <Card key={entry.version}>
              <div className="flex items-center gap-3">
                <p className="font-display text-lg text-porcelain">v{entry.version}</p>
                {i === 0 && <Tag tone="gold">{zh ? "最新" : "Latest"}</Tag>}
              </div>
              <p className="mt-1 text-xs text-mist">{entry.date}</p>
              <ul className="mt-3 space-y-2">
                {entry.notes.map((n) => (
                  <li key={n} className="flex items-start gap-2 text-sm leading-relaxed text-mist">
                    <span className="text-gold">·</span>
                    <span>{n}</span>
                  </li>
                ))}
              </ul>
            </Card>
          ))}
        </div>
      </Section>

      {/* Support, legal, social */}
      <div className="border-t border-hairline bg-obsidian/40">
        <Section className="py-12 sm:py-16">
          <Meridian className="mb-10" />
          <div className="grid gap-8 lg:grid-cols-3">
            <div>
              <p className="font-display text-lg text-porcelain">{zh ? "隐私与条款" : "Privacy and terms"}</p>
              <ul className="mt-3 space-y-2 text-sm">
                <li>
                  <Link href="/legal/app" className="text-mist transition-colors hover:text-gold">
                    {zh ? "App 隐私声明 →" : "App Privacy Notice →"}
                  </Link>
                </li>
                <li>
                  <Link href="/legal/terms" className="text-mist transition-colors hover:text-gold">
                    {zh ? "用户协议 →" : "User Terms →"}
                  </Link>
                </li>
                <li>
                  <Link href="/legal/privacy" className="text-mist transition-colors hover:text-gold">
                    {zh ? "隐私政策 →" : "Privacy Policy →"}
                  </Link>
                </li>
              </ul>
            </div>
            <div>
              <p className="font-display text-lg text-porcelain">{zh ? "客服与支持" : "Customer service"}</p>
              <p className="mt-3 text-sm leading-relaxed text-mist">
                {zh
                  ? "人工礼宾工作日 10:00–19:00（CST）在线；也可通过微信公众号留言。"
                  : "A human concierge is available on business days, 10:00–19:00 CST — or leave a message via the WeChat Official Account."}
              </p>
              <ul className="mt-3 space-y-2 text-sm">
                <li>
                  <Link href="/legal/contact" className="text-mist transition-colors hover:text-gold">
                    {zh ? "联系我们 →" : "Contact us →"}
                  </Link>
                </li>
                <li>
                  <Link href="/wechat" className="text-mist transition-colors hover:text-gold">
                    {zh ? "微信公众号客服 →" : "WeChat customer service →"}
                  </Link>
                </li>
              </ul>
            </div>
            <div>
              <p className="font-display text-lg text-porcelain">{zh ? "关注官方社媒" : "Follow ZOTAIX"}</p>
              <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2">
                {social.map((s) => (
                  <a
                    key={s.id}
                    href={socialHref(s)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-mist transition-colors hover:text-gold"
                  >
                    {s.platform} ↗
                  </a>
                ))}
              </div>
              <p className="mt-3 text-xs text-mist">
                <Link href="/social" className="transition-colors hover:text-gold">
                  {zh ? "查看全球社媒矩阵 →" : "See the global social matrix →"}
                </Link>
              </p>
            </div>
          </div>
        </Section>
      </div>
    </>
  );
}
