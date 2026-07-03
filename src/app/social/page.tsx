import Link from "next/link";
import type { Metadata } from "next";
import { ButtonLink, Card, Meridian, Notice, PageHero, Section, SectionHeader, Tag } from "@/components/ui";
import { brand } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";
import type { SocialAccount } from "@/lib/types";
import SubscribeForm from "./SubscribeForm";

export const metadata: Metadata = pageMetadata({
  title: "ZOTAIX on every platform — the global social matrix",
  description:
    "Official ZOTAIX accounts on Instagram, TikTok, X, YouTube, LinkedIn, and more: Reserve stories, label POVs, atelier process, enterprise collaboration, and the overseas brand letter.",
  path: "/social",
  keywords: ["ZOTAIX", "social media", "Instagram", "TikTok", "YouTube", "LinkedIn"],
});

const monograms: Record<string, string> = {
  instagram: "IG",
  tiktok: "TT",
  x: "X",
  youtube: "YT",
  linkedin: "in",
  facebook: "f",
  pinterest: "P",
  threads: "@",
};

function accountHref(account: SocialAccount): string {
  if (!account.tracking_params) return account.official_url;
  return `${account.official_url}${account.official_url.includes("?") ? "&" : "?"}${account.tracking_params}`;
}

function handleOf(account: SocialAccount): string {
  return account.official_url.replace(/^https?:\/\//, "");
}

export default async function SocialPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const accounts = data.social_accounts
    .filter((a) => a.enabled)
    .sort((a, b) => a.display_order - b.display_order);
  const linkedin = accounts.find((a) => a.icon === "linkedin");

  const pillars = [
    {
      platform: "Instagram",
      icon: "IG",
      title: zh ? "档案故事 · Reserve stories" : "Reserve stories",
      desc: zh
        ? "证书走查、瓶身静物与档案美学：每一条被保存的记录，都值得一条 60 秒的讲述。"
        : "Certificate walkthroughs, bottle stills, and archive aesthetics — every kept record earns a sixty-second telling, from emotion to QR.",
      tag: zh ? "静物 · 走查" : "Stills · walkthroughs",
    },
    {
      platform: "TikTok",
      icon: "TT",
      title: zh ? "标签视角 · Label POVs" : "Label POVs",
      desc: zh
        ? "「你的失恋有了一张标签」——从一句话到可分享标签的情绪补给内容，合拍友好。"
        : "“POV: your breakup gets a label” — emotional supply energy, from one sentence to a shareable label, in duet-friendly formats.",
      tag: zh ? "情绪补给 · 合拍" : "Supply · duets",
    },
    {
      platform: "YouTube",
      icon: "YT",
      title: zh ? "工坊过程 · Atelier process" : "Atelier process",
      desc: zh
        ? "长视频记录从提案到设计版本再到人工确认的全过程，含企业案例深访。"
        : "Long-form process films: proposal to design versions to human confirmation, plus deep dives into enterprise cases.",
      tag: zh ? "长视频 · 案例" : "Long-form · cases",
    },
  ];

  return (
    <>
      <PageHero
        eyebrow={zh ? "全球社媒矩阵 · Global social" : "Global social matrix"}
        title={zh ? "同一个卓序，出现在每个平台" : "One ZOTAIX, present on every platform"}
        description={
          zh
            ? "官方账号覆盖 Instagram、TikTok、X、YouTube、LinkedIn 等平台：档案故事、标签视角与工坊过程按同一份内容日历发布，企业合作与海外订阅由此进入。"
            : "Official accounts span Instagram, TikTok, X, YouTube, LinkedIn, and more. Reserve stories, label POVs, and atelier process ship on one shared content calendar — with enterprise collaboration and the overseas letter starting here."
        }
      >
        <ButtonLink href="#accounts" variant="gold">
          {zh ? "查看官方账号" : "Browse official accounts"}
        </ButtonLink>
        <ButtonLink href="#newsletter" variant="outline">
          {zh ? "订阅品牌通讯" : "Subscribe to the letter"}
        </ButtonLink>
      </PageHero>

      {/* Brand sentence, large */}
      <div className="border-b border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <p className="text-xs font-semibold uppercase tracking-[0.25em] text-gold">
            {zh ? "一句话说清卓序" : "The brand in one sentence"}
          </p>
          <p className="font-display mt-5 max-w-4xl text-xl leading-relaxed text-porcelain sm:text-3xl sm:leading-snug">
            “{brand.en}”
          </p>
          <p className="mt-5 max-w-3xl text-sm leading-relaxed text-mist">{brand.zh}</p>
        </Section>
      </div>

      {/* Platform cards */}
      <Section id="accounts" className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "官方账号" : "Official accounts"}
          title={zh ? "认准这些主页" : "The accounts to follow"}
          description={
            zh
              ? "以下为全部官方账号，链接均带来源参数以便团队衡量各平台表现。"
              : "Every official account, listed in order. Links carry source parameters so the team can measure each platform’s pull."
          }
        />
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {accounts.map((a) => (
            <a key={a.id} href={accountHref(a)} target="_blank" rel="noopener noreferrer" className="block h-full">
              <Card hover className="flex h-full flex-col">
                <div className="flex items-center gap-3">
                  <span className="font-display flex h-10 w-10 items-center justify-center rounded-full border border-gold/40 text-sm text-gold">
                    {monograms[a.icon] ?? a.platform.slice(0, 2).toUpperCase()}
                  </span>
                  <p className="font-display text-base text-porcelain">{a.platform}</p>
                </div>
                <p className="mt-3 flex-1 break-all text-xs text-mist">{handleOf(a)}</p>
                <p className="mt-3 text-xs text-gold">{zh ? "打开主页 ↗" : "Open profile ↗"}</p>
              </Card>
            </a>
          ))}
        </div>
        <div className="mt-6">
          <Notice tone="gold" title={zh ? "一份日历，驱动全部平台" : "One calendar drives every platform"}>
            {zh
              ? "所有平台的选题、排期与负责人由 ZOTAIX 管理后台的内容日历统一管理——同一个故事在各平台以最合适的形式出现。"
              : "Topics, schedules, and owners for every platform are managed in one content calendar inside the ZOTAIX admin console — the same story appears on each platform in its best-fitting form."}
          </Notice>
        </div>
      </Section>

      {/* Content pillars */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "内容支柱" : "Content pillars"}
            title={zh ? "三条内容线，一条定制链" : "Three content lines, one customization chain"}
            description={
              zh
                ? "每个平台承担定制链的一段：Instagram 讲档案，TikTok 讲情绪，YouTube 讲过程。"
                : "Each platform carries one stretch of the chain: Instagram tells the archive, TikTok tells the emotion, YouTube tells the process."
            }
          />
          <div className="mt-8 grid gap-4 lg:grid-cols-3">
            {pillars.map((p) => (
              <Card key={p.platform} className="flex h-full flex-col">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <span className="font-display flex h-9 w-9 items-center justify-center rounded-full border border-gold/40 text-xs text-gold">
                      {p.icon}
                    </span>
                    <p className="font-display text-base text-porcelain">{p.platform}</p>
                  </div>
                  <Tag tone="gold">{p.tag}</Tag>
                </div>
                <p className="font-display mt-4 text-lg text-porcelain">{p.title}</p>
                <p className="mt-2 flex-1 text-sm leading-relaxed text-mist">{p.desc}</p>
              </Card>
            ))}
          </div>
        </Section>
      </div>

      {/* LinkedIn + newsletter */}
      <Section id="newsletter" className="py-14 sm:py-20">
        <div className="grid gap-10 lg:grid-cols-2">
          <div>
            <SectionHeader
              eyebrow="LinkedIn"
              title={zh ? "企业合作，从一次连接开始" : "Enterprise collaboration starts with a connection"}
              description={
                zh
                  ? "LinkedIn 官方主页发布企业礼赠案例与合作方法论；品牌联名、酒店与文旅项目请走高定合作入口，人工礼宾一个工作日内回复。"
                  : "The LinkedIn page publishes enterprise gifting cases and collaboration playbooks. For brand collaborations, hotels, and cultural tourism projects, use the Maison collaboration entry — a human concierge replies within one business day."
              }
            />
            <div className="mt-6 flex flex-wrap gap-3">
              <ButtonLink href="/maison#collaboration" variant="gold">
                {zh ? "发起企业合作" : "Start a collaboration"}
              </ButtonLink>
              {linkedin && (
                <a
                  href={accountHref(linkedin)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center justify-center gap-2 rounded-md border border-hairline px-5 py-2.5 text-sm font-medium text-porcelain transition-colors hover:border-gold hover:text-gold"
                >
                  {zh ? "LinkedIn 主页 ↗" : "LinkedIn page ↗"}
                </a>
              )}
              <ButtonLink href="/cases" variant="ghost">
                {zh ? "查看企业案例 →" : "See enterprise cases →"}
              </ButtonLink>
            </div>
          </div>
          <div>
            <SectionHeader
              eyebrow={zh ? "海外订阅 · Brand letter" : "Overseas subscription"}
              title={zh ? "把卓序寄进你的收件箱" : "The ZOTAIX letter, in your inbox"}
              description={
                zh
                  ? "面向海外读者的英文品牌通讯：品牌志精选、共创进展与档案故事。留下邮箱，订阅即刻记录。"
                  : "An English-language letter for overseas readers: Journal highlights, co-creation progress, and Reserve stories. Leave your email and the subscription is recorded immediately."
              }
            />
            <div className="mt-6">
              <SubscribeForm zh={zh} />
            </div>
          </div>
        </div>
      </Section>

      {/* Share-ready note */}
      <div className="border-t border-hairline bg-obsidian/40">
        <Section className="py-12 sm:py-16">
          <Meridian className="mb-10" />
          <div className="grid gap-8 lg:grid-cols-[1fr_auto]">
            <div>
              <p className="font-display text-lg text-porcelain">
                {zh ? "每个页面都是可分享的" : "Every page ships share-ready"}
              </p>
              <p className="mt-3 max-w-2xl text-sm leading-relaxed text-mist">
                {zh
                  ? "平台的每个公开页面都输出 og:title、og:description、og:image、Twitter 卡片与规范链接——链接贴到任何平台都会自动展开为品牌卡片，公开的档案页与共创页也不例外。"
                  : "Every public page on the platform emits og:title, og:description, og:image, a Twitter summary card, and a canonical URL — paste a link on any platform and it unfurls as a branded card, public Reserve and co-creation pages included."}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-3 lg:flex-col lg:items-end">
              <ButtonLink href="/wechat" variant="outline">
                {zh ? "微信公众号 →" : "WeChat Official Account →"}
              </ButtonLink>
              <ButtonLink href="/download" variant="outline">
                {zh ? "下载 ZOTAIX App →" : "Download the app →"}
              </ButtonLink>
            </div>
          </div>
          <p className="mt-8 text-xs text-mist">
            {zh ? "内容合规：" : "Content compliance: "}
            <Link href="/legal/ai" className="transition-colors hover:text-gold">
              {zh ? "AI 内容声明" : "AI Content Notice"}
            </Link>
            {" · "}
            <Link href="/legal/alcohol" className="transition-colors hover:text-gold">
              {zh ? "酒类合规" : "Alcohol Compliance"}
            </Link>
            {" · "}
            <Link href="/legal/minors" className="transition-colors hover:text-gold">
              {zh ? "未成年人保护" : "Minor Protection"}
            </Link>
          </p>
        </Section>
      </div>
    </>
  );
}
