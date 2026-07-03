import Link from "next/link";
import type { Metadata } from "next";
import { ButtonLink, Card, Meridian, Notice, ProgressBar, Section, SectionHeader, Tag } from "@/components/ui";
import { getLocale, tt } from "@/lib/locale";
import { headline, subheadline, complianceNotice } from "@/lib/copy";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Turn emotions into collectible spirits, fragrances, bottles, and gifts",
  description:
    "ZOTAIX uses AI to understand your state, recipient, scenario, and budget — then generates liquid directions, fragrance directions, bottle visuals, label copy, gift-box stories, and digital identity records.",
  path: "/",
});

export default async function HomePage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const announcement = data.cms_blocks.find((b) => b.key === "home.announcement" && b.enabled);
  const heroBadge = data.cms_blocks.find((b) => b.key === "home.hero.badge" && b.enabled);
  const projects = data.co_creation_projects.filter((p) => p.public_visible && p.review_status === "approved").slice(0, 2);
  const featuredCases = data.case_studies.filter((c) => c.featured).slice(0, 2);
  const publicReserve = data.reserve_records.filter((r) => r.privacy_level === "public").slice(0, 3);

  const primaryCtas = [
    { href: "/concierge", en: "Start AI Concierge", zh: "启动 AI 礼宾", variant: "gold" as const },
    { href: "/supply", en: "Generate an Emotional Supply", zh: "生成情绪补给", variant: "outline" as const },
    { href: "/maison", en: "Customize a Premium Gift", zh: "定制高端礼物", variant: "outline" as const },
    { href: "/co-create", en: "Enter Co-Creation Pool", zh: "进入共创池", variant: "outline" as const },
    { href: "/reserve", en: "View Reserve Records", zh: "查看档案记录", variant: "outline" as const },
    { href: "/download", en: "Download ZOTAIX App", zh: "下载 ZOTAIX App", variant: "outline" as const },
  ];

  const modules = [
    { href: "/concierge", en: "AI Daily Concierge", zh: "AI 日常礼宾", desc: zh ? "随口聊聊今天，AI 用关键词与轻建议回应你。" : "Talk about today; the AI answers with keywords and a light suggestion.", icon: "◐" },
    { href: "/forge", en: "AI Emotional Spirit", zh: "AI 情绪之酒", desc: zh ? "把一种情绪变成酒体方向、命名与瓶身文案。" : "Turn one emotion into a liquid direction, names, and label copy.", icon: "◈" },
    { href: "/forge?mode=fragrance", en: "AI Fragrance Profile", zh: "AI 香氛画像", desc: zh ? "从场景与偏好生成专属香氛方向。" : "Generate a fragrance direction from scenario and preference.", icon: "❖" },
    { href: "/studio", en: "AI Bottle Design", zh: "AI 瓶身设计", desc: zh ? "预览瓶身、标签、包装与情绪卡片。" : "Preview bottles, labels, packaging, and emotional cards.", icon: "◇" },
    { href: "/supply", en: "ZOTAIX Supply", zh: "情绪补给线", desc: zh ? "轻量、俏皮、可分享的情绪补给对象。" : "Light, playful, shareable emotional supply objects.", icon: "✶" },
    { href: "/maison", en: "Maison ZOTAIX", zh: "高定礼赠线", desc: zh ? "企业礼赠与私人高定，人工礼宾全程确认。" : "Enterprise gifting and private bespoke with human concierge.", icon: "◆" },
    { href: "/co-create", en: "Co-Creation Pool", zh: "共创铸造池", desc: zh ? "十人即可开启公开共创页，百瓶解锁风味评审。" : "10 people open a public page; 100 bottles unlock flavor review.", icon: "⬡" },
    { href: "/reserve", en: "Reserve Records", zh: "档案馆", desc: zh ? "每个对象都有 ZOTAIX ID、证书与补铸入口。" : "Every object gets a ZOTAIX ID, certificate, and replenishment entry.", icon: "▣" },
    { href: "/download", en: "App Download", zh: "App 下载", desc: zh ? "PWA 安装与 iOS / Android 应用入口。" : "PWA install plus iOS / Android app entries.", icon: "▤" },
    { href: "/wechat", en: "WeChat Official Account", zh: "微信公众号", desc: zh ? "关注公众号，微信内直达礼宾与共创。" : "Follow for concierge and co-creation inside WeChat.", icon: "◉" },
    { href: "/social", en: "Global Social Media", zh: "全球社媒矩阵", desc: zh ? "Instagram、TikTok、X、YouTube、LinkedIn 官方账号。" : "Official Instagram, TikTok, X, YouTube, LinkedIn accounts.", icon: "✳" },
  ];

  const steps = [
    { en: "Tell the concierge your state", zh: "告诉礼宾你的状态", desc: zh ? "情绪、对象、场景、预算——一句话也可以。" : "Emotion, recipient, scenario, budget — one sentence is enough." },
    { en: "Receive a structured object", zh: "获得结构化对象", desc: zh ? "酒体方向、香氛方向、命名、文案与数字印记。" : "Liquid direction, fragrance direction, names, copy, digital mark." },
    { en: "Save it to your archive", zh: "存入你的档案", desc: zh ? "先创造并保存对象，再决定它成为什么。" : "Create and save the object first; decide what it becomes later." },
    { en: "Make it real — if you choose", zh: "让它成真 —— 如果你愿意", desc: zh ? "实体铸造、共创、企业礼赠均经人工确认与报价。" : "Casting, co-creation, and enterprise gifting go through human confirmation and quotes." },
  ];

  return (
    <>
      {announcement && (
        <div className="border-b border-gold/20 bg-gold/5">
          <Section className="flex items-center gap-3 py-2.5">
            <span className="text-xs text-gold">◈</span>
            <p className="truncate text-xs text-mist sm:text-sm">{announcement.content}</p>
            <Link href="/co-create" className="ml-auto shrink-0 text-xs text-gold hover:underline">
              {zh ? "查看 →" : "View →"}
            </Link>
          </Section>
        </div>
      )}

      {/* Hero */}
      <div className="zx-grid-bg border-b border-hairline">
        <Section className="py-16 sm:py-24">
          {heroBadge && (
            <p className="mb-4 inline-block rounded-full border border-gold/40 px-3 py-1 text-xs uppercase tracking-[0.25em] text-gold">
              {heroBadge.content}
            </p>
          )}
          <h1 className="font-display max-w-4xl text-3xl leading-tight text-porcelain sm:text-5xl lg:text-6xl">
            {tt(locale, headline.en, headline.zh)}
          </h1>
          <p className="mt-6 max-w-3xl text-sm leading-relaxed text-mist sm:text-lg">
            {tt(locale, subheadline.en, subheadline.zh)}
          </p>
          <div className="mt-9 flex flex-wrap gap-3">
            {primaryCtas.map((cta) => (
              <ButtonLink key={cta.href} href={cta.href} variant={cta.variant}>
                {zh ? cta.zh : cta.en}
              </ButtonLink>
            ))}
          </div>
        </Section>
      </div>

      {/* What you can do */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "你可以做什么" : "What you can do"}
          title={zh ? "一条 AI 定制链，从情绪到实物" : "One AI customization chain, from emotion to object"}
          description={
            zh
              ? "先创造并保存一个属于你的对象——之后再决定把它变成真实的酒、香氛、瓶身、礼盒、企业礼赠、共创项目或高端礼宾委托。"
              : "First create and save a personalized object — then decide whether it becomes a real spirit, fragrance, bottle, gift box, enterprise gift, co-creation project, or high-end concierge request."
          }
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {modules.map((m) => (
            <Link key={m.href + m.en} href={m.href}>
              <Card hover className="h-full">
                <div className="flex items-start gap-3">
                  <span className="text-lg text-gold">{m.icon}</span>
                  <div>
                    <p className="font-display text-base text-porcelain">{zh ? m.zh : m.en}</p>
                    <p className="mt-1.5 text-sm leading-relaxed text-mist">{m.desc}</p>
                  </div>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      </Section>

      {/* Dual line */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "双产品线" : "Dual product structure"}
            title={zh ? "同一条定制链，两种交付深度" : "The same chain, two depths of delivery"}
          />
          <div className="mt-10 grid gap-5 lg:grid-cols-2">
            <Card className="border-gold/25">
              <p className="text-xs font-semibold uppercase tracking-[0.25em] text-gold">Maison ZOTAIX</p>
              <p className="font-display mt-2 text-xl text-porcelain">{zh ? "高定礼赠线" : "The premium line"}</p>
              <p className="mt-3 text-sm leading-relaxed text-mist">
                {zh
                  ? "企业礼赠、客户答谢、高端宴席、私人庆典、品牌联名、城市伴手礼。AI 礼宾 + 人工确认 + 高级设计 + 报价交付 + 售后与档案身份。"
                  : "Enterprise gifting, client appreciation, banquets, private celebrations, brand collaborations, city souvenirs. AI concierge + human confirmation + premium design + quotation + delivery + aftercare + Reserve identity."}
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <Tag tone="gold">{zh ? "人工礼宾" : "Human concierge"}</Tag>
                <Tag tone="gold">{zh ? "企业定制" : "Enterprise"}</Tag>
                <Tag tone="gold">{zh ? "报价交付" : "Quotation"}</Tag>
              </div>
              <div className="mt-5">
                <ButtonLink href="/maison" variant="gold">{zh ? "进入 Maison" : "Enter Maison"}</ButtonLink>
              </div>
            </Card>
            <Card className="border-supply/25">
              <p className="text-xs font-semibold uppercase tracking-[0.25em] text-supply">ZOTAIX Supply</p>
              <p className="font-display mt-2 text-xl text-porcelain">{zh ? "情绪补给线" : "The emotional supply line"}</p>
              <p className="mt-3 text-sm leading-relaxed text-mist">
                {zh
                  ? "生日、失恋恢复、考试解压、职场情绪、友情与恋人、派对场景。AI 懂我 + 个性标签 + 情绪命名 + 数字徽章 + 可分享档案 + 低门槛共创。"
                  : "Birthdays, breakup recovery, exam stress, workplace feelings, friends and partners, parties. AI understands me + personalized labels + emotional naming + digital badges + shareable archives + low-barrier co-creation."}
              </p>
              <div className="mt-4 flex flex-wrap gap-2">
                <Tag tone="supply">{zh ? "情绪卡片" : "Emotional cards"}</Tag>
                <Tag tone="supply">{zh ? "低酒精 / 零酒精" : "Low / zero proof"}</Tag>
                <Tag tone="supply">{zh ? "共创" : "Co-creation"}</Tag>
              </div>
              <div className="mt-5">
                <ButtonLink href="/supply" variant="supply">{zh ? "进入 Supply" : "Enter Supply"}</ButtonLink>
              </div>
            </Card>
          </div>
        </Section>
      </div>

      {/* How it works */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "为什么 AI 重要" : "Why AI matters"}
          title={zh ? "四步：从被理解，到被保存" : "Four steps: from being understood to being kept"}
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {steps.map((s, i) => (
            <Card key={s.en}>
              <p className="font-display text-3xl text-gold/60">{String(i + 1).padStart(2, "0")}</p>
              <p className="font-display mt-2 text-base text-porcelain">{zh ? s.zh : s.en}</p>
              <p className="mt-2 text-sm leading-relaxed text-mist">{s.desc}</p>
            </Card>
          ))}
        </div>
        <div className="mt-8">
          <Notice tone="gold">{tt(locale, complianceNotice.en, complianceNotice.zh)}</Notice>
        </div>
      </Section>

      {/* Live co-creation */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <SectionHeader
              eyebrow={zh ? "正在发生" : "Happening now"}
              title={zh ? "共创池 · 集结中的项目" : "Co-creation pool · projects gathering"}
            />
            <ButtonLink href="/co-create" variant="outline">{zh ? "查看全部" : "View all"}</ButtonLink>
          </div>
          <div className="mt-8 grid gap-4 lg:grid-cols-2">
            {projects.map((p) => (
              <Link key={p.id} href={`/co-create/${p.id}`}>
                <Card hover className="h-full">
                  <div className="flex flex-wrap gap-2">
                    {p.emotion_tags.map((t) => (
                      <Tag key={t}>{t}</Tag>
                    ))}
                  </div>
                  <p className="font-display mt-3 text-lg text-porcelain">{p.title}</p>
                  <p className="mt-2 line-clamp-2 text-sm text-mist">{p.concept}</p>
                  <div className="mt-4 space-y-1.5">
                    <div className="flex justify-between text-xs text-mist">
                      <span>
                        {p.current_quantity}/{p.target_quantity} {zh ? "已预订" : "reserved"}
                      </span>
                      <span>{p.supporters} {zh ? "位支持者" : "supporters"}</span>
                    </div>
                    <ProgressBar value={p.current_quantity} max={p.target_quantity} />
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        </Section>
      </div>

      {/* Reserve + cases */}
      <Section className="py-14 sm:py-20">
        <div className="grid gap-10 lg:grid-cols-2">
          <div>
            <SectionHeader
              eyebrow="Reserve"
              title={zh ? "被保存的对象" : "Objects that were kept"}
              description={zh ? "公开档案页可被扫码访问 —— 每一条记录都是一个被选择保存的时刻。" : "Public archive pages open from a QR scan — every record is a moment someone chose to keep."}
            />
            <div className="mt-6 space-y-3">
              {publicReserve.map((r) => (
                <Link key={r.id} href={`/reserve/${r.id}`} className="block">
                  <Card hover className="flex items-center justify-between gap-4 !p-4">
                    <div>
                      <p className="font-display text-sm text-porcelain">{r.object_name}</p>
                      <p className="mt-0.5 text-xs text-mist">{r.zotaix_id} · {r.object_type}</p>
                    </div>
                    <span className="text-gold">→</span>
                  </Card>
                </Link>
              ))}
            </div>
          </div>
          <div>
            <SectionHeader
              eyebrow={zh ? "案例" : "Cases"}
              title={zh ? "从一句话到一次交付" : "From one sentence to one delivery"}
            />
            <div className="mt-6 space-y-3">
              {featuredCases.map((c) => (
                <Link key={c.id} href={`/cases/${c.slug}`} className="block">
                  <Card hover className="!p-4">
                    <p className="text-xs uppercase tracking-wider text-gold">{c.category}</p>
                    <p className="font-display mt-1 text-sm text-porcelain">{zh ? c.title_zh : c.title}</p>
                    <p className="mt-1.5 line-clamp-2 text-xs leading-relaxed text-mist">{c.summary}</p>
                  </Card>
                </Link>
              ))}
              <ButtonLink href="/cases" variant="ghost" className="!px-0">
                {zh ? "全部案例 →" : "All cases →"}
              </ButtonLink>
            </div>
          </div>
        </div>
      </Section>

      {/* Channels */}
      <div className="border-t border-hairline bg-obsidian/40">
        <Section className="py-12">
          <Meridian className="mb-10" />
          <div className="grid gap-6 text-center sm:grid-cols-3">
            {[
              { href: "/download", title: zh ? "ZOTAIX App" : "ZOTAIX App", desc: zh ? "PWA 安装 · iOS · Android" : "PWA install · iOS · Android" },
              { href: "/wechat", title: zh ? "微信公众号" : "WeChat Official Account", desc: zh ? "关注卓序，微信内直达礼宾" : "Follow ZOTAIX 卓序 inside WeChat" },
              { href: "/social", title: zh ? "全球社媒" : "Global Social", desc: "Instagram · TikTok · X · YouTube · LinkedIn" },
            ].map((c) => (
              <Link key={c.href} href={c.href} className="group">
                <p className="font-display text-lg text-porcelain transition-colors group-hover:text-gold">{c.title}</p>
                <p className="mt-1 text-sm text-mist">{c.desc}</p>
              </Link>
            ))}
          </div>
        </Section>
      </div>
    </>
  );
}
