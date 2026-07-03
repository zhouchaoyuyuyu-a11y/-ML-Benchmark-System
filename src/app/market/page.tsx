import Link from "next/link";
import type { Metadata } from "next";
import { ButtonLink, Card, Notice, PageHero, ProgressBar, Section, SectionHeader, Stat, Tag } from "@/components/ui";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Creative Market — public proposals, founder rights, and open archives",
  description:
    "Browse public creative proposals, label concepts, bottle visuals, co-creation founder rights, digital badges, and public Reserve pages. Request authorization, apply to collaborate, or open an enterprise lead — all physical execution passes human review.",
  path: "/market",
  keywords: ["creative market", "design authorization", "founder rights", "public archive", "label concepts"],
});

type BadgeTone = "default" | "gold" | "supply" | "jade" | "ember";

interface Listing {
  id: string;
  badge: { en: string; zh: string; tone: BadgeTone };
  title: string;
  description: string;
  tags: string[];
  meta: string;
  progress?: { value: number; max: number };
  created_at: string;
}

export default async function MarketPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();

  const publicDrafts = data.object_drafts.filter((d) => d.public_visible);
  const publicProjects = data.co_creation_projects.filter((p) => p.public_visible && p.review_status === "approved");
  const publicRecords = data.reserve_records.filter((r) => r.privacy_level === "public");

  const listings: Listing[] = [
    ...publicDrafts.map((d): Listing => ({
      id: d.id,
      badge: { en: "Creative proposal", zh: "创意提案", tone: "default" },
      title: d.title,
      description: d.label_copy ?? d.liquid_direction ?? d.scent_direction ?? d.visual_style ?? d.scene ?? "",
      tags: d.emotion_tags,
      meta: d.object_type.replace(/_/g, " "),
      created_at: d.created_at,
    })),
    ...publicProjects.map((p): Listing => ({
      id: p.id,
      badge: { en: "Founder rights / participation", zh: "发起人权益 / 参与", tone: "gold" },
      title: p.title,
      description: p.concept,
      tags: p.emotion_tags,
      meta: `${p.supporters} ${zh ? "位支持者" : "supporters"}`,
      progress: { value: p.current_quantity, max: p.target_quantity },
      created_at: p.created_at,
    })),
    ...publicRecords.map((r): Listing => ({
      id: r.id,
      badge: { en: "Public archive", zh: "公开档案", tone: "jade" },
      title: r.object_name,
      description: r.label_copy ?? r.product_direction ?? r.relationship_scene ?? "",
      tags: r.emotion_tags,
      meta: r.zotaix_id,
      created_at: r.created_at,
    })),
  ].sort((a, b) => (a.created_at < b.created_at ? 1 : -1));

  const rights: { icon: string; title: { en: string; zh: string }; body: { en: string; zh: string } }[] = [
    {
      icon: "❖",
      title: { en: "Authorization ≠ ownership transfer", zh: "授权 ≠ 所有权转移" },
      body: {
        en: "An authorization grants a scoped, reviewed right to use a design commercially. The creator keeps ownership of the work and its Reserve identity.",
        zh: "授权是一项经审核、限定范围的商用使用权。作品的所有权与档案身份始终属于创作者。",
      },
    },
    {
      icon: "◆",
      title: { en: "Platform review required", zh: "必须经平台审核" },
      body: {
        en: "Every authorization, collaboration, and physical execution passes human review and compliance checks before anything is produced or sold.",
        zh: "每一次授权、合作与实体执行，都在生产或销售前经过人工审核与合规审查。",
      },
    },
    {
      icon: "▣",
      title: { en: "Designer income share", zh: "设计师收益分成" },
      body: {
        en: "When a design is commercially used under an approved authorization, its creator receives an agreed share of the proceeds — the share is set per agreement during human review.",
        zh: "设计在经批准的授权下被商用时，创作者按约定获得收益分成——分成比例在人工审核时按协议逐案确定。",
      },
    },
  ];

  return (
    <>
      <PageHero
        eyebrow={zh ? "创意市场" : "Creative Market"}
        title={zh ? "公开的提案、权益与档案，都在这里陈列" : "Public proposals, rights, and archives, on open display"}
        description={
          zh
            ? "创意市场陈列平台上公开的一切：创意提案、标签概念、瓶身视觉、共创发起人权益与名额、数字徽章、公开档案页。你可以申请设计授权、加入共创、提交合作申请或开启企业线索——设计师在授权商用中获得收益分成。"
            : "The Creative Market displays everything public on the platform: creative proposals, label concepts, bottle visuals, co-creation founder rights and slots, digital badges, and public Reserve pages. Request design authorization, join a co-creation, apply to collaborate, or open an enterprise lead — designers earn an income share on authorized commercial use."
        }
      >
        <ButtonLink href="#listings" variant="gold">
          {zh ? "浏览陈列" : "Browse listings"}
        </ButtonLink>
        <ButtonLink href="/forge" variant="outline">
          {zh ? "创造并公开你的提案" : "Create and publish your own"}
        </ButtonLink>
        <ButtonLink href="/trade" variant="outline">
          {zh ? "报价与授权通道" : "Quotes & rights channel"}
        </ButtonLink>
      </PageHero>

      {/* Stats */}
      <Section className="py-10">
        <div className="grid gap-4 sm:grid-cols-3">
          <Stat
            label={zh ? "公开创意提案" : "Public creative proposals"}
            value={String(publicDrafts.length)}
            hint={zh ? "标签概念、瓶身视觉与命名" : "Label concepts, bottle visuals, naming"}
          />
          <Stat
            label={zh ? "开放中的共创权益" : "Open founder-rights runs"}
            value={String(publicProjects.length)}
            hint={zh ? "发起人名额与参与席位" : "Founder slots and participation seats"}
          />
          <Stat
            label={zh ? "公开档案页" : "Public archive pages"}
            value={String(publicRecords.length)}
            hint={zh ? "可扫码访问的 Reserve 记录" : "QR-reachable Reserve records"}
          />
        </div>
      </Section>

      {/* Listings */}
      <Section id="listings" className="py-8 sm:py-12">
        <SectionHeader
          eyebrow={zh ? "陈列" : "Listings"}
          title={zh ? "正在公开陈列的对象" : "Objects currently on display"}
          description={
            zh
              ? "三类陈列：创意提案可申请授权，共创项目可加入或成为发起人，公开档案可作为补铸与合作的参照。"
              : "Three kinds of listings: creative proposals open to authorization, co-creation runs open to joining or founding, and public archives that anchor replenishment and collaboration."
          }
        />
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {listings.map((l) => (
            <Link key={l.id} href={`/market/${l.id}`} className="block h-full">
              <Card hover className="flex h-full flex-col">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <Tag tone={l.badge.tone}>{zh ? l.badge.zh : l.badge.en}</Tag>
                  <span className="text-xs text-mist">{l.meta}</span>
                </div>
                <p className="font-display mt-3 text-lg leading-snug text-porcelain">{l.title}</p>
                {l.description && (
                  <p className="mt-2 line-clamp-3 text-sm leading-relaxed text-mist">{l.description}</p>
                )}
                {l.progress && (
                  <div className="mt-4 space-y-1.5">
                    <div className="flex justify-between text-xs text-mist">
                      <span>
                        {l.progress.value}/{l.progress.max} {zh ? "已预订" : "reserved"}
                      </span>
                    </div>
                    <ProgressBar value={l.progress.value} max={l.progress.max} />
                  </div>
                )}
                {l.tags.length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {l.tags.map((t) => (
                      <Tag key={t}>{t}</Tag>
                    ))}
                  </div>
                )}
                <span className="mt-auto pt-4 text-xs text-gold">{zh ? "查看陈列 →" : "View listing →"}</span>
              </Card>
            </Link>
          ))}
        </div>
      </Section>

      {/* Rights explainer */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "权益说明" : "How rights work"}
            title={zh ? "三条规则，保护创作者与买方" : "Three rules that protect creators and buyers"}
          />
          <div className="mt-8 grid gap-4 lg:grid-cols-3">
            {rights.map((r) => (
              <Card key={r.title.en} className="h-full">
                <span className="text-lg text-gold">{r.icon}</span>
                <p className="font-display mt-2 text-base text-porcelain">{zh ? r.title.zh : r.title.en}</p>
                <p className="mt-2 text-sm leading-relaxed text-mist">{zh ? r.body.zh : r.body.en}</p>
              </Card>
            ))}
          </div>
          <div className="mt-8">
            <Notice tone="ember" title={zh ? "风险提示" : "Risk notice"}>
              {zh
                ? "创意市场不进行酒类转售：陈列的是创意、权益与档案，而不是可交易的酒。所有实体执行——铸造、授权商用、企业礼赠——都必须经过平台人工审核与合规审查。"
                : "The Creative Market does not resell alcohol: listings are creative work, rights, and archives — not tradable bottles. All physical execution — casting, authorized commercial use, enterprise gifting — must pass platform human review and compliance checks."}
            </Notice>
          </div>
        </Section>
      </div>

      {/* Collaboration + enterprise */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "更进一步" : "Go further"}
          title={zh ? "合作申请与企业线索" : "Collaboration applications and enterprise leads"}
        />
        <div className="mt-8 grid gap-5 lg:grid-cols-2">
          <Card className="border-gold/25">
            <p className="font-display text-lg text-porcelain">{zh ? "以创作者身份合作" : "Collaborate as a creator"}</p>
            <p className="mt-2 text-sm leading-relaxed text-mist">
              {zh
                ? "设计师、工作室与品牌方可以就市场上的任何陈列提交合作申请——授权、联名或共同开发。礼宾团队逐案审核范围与分成协议。"
                : "Designers, studios, and brands can apply to collaborate on any listing — authorization, co-branding, or joint development. The concierge team reviews scope and income-share agreements case by case."}
            </p>
            <div className="mt-4">
              <ButtonLink href="/trade" variant="gold">
                {zh ? "提交合作申请" : "Submit a collaboration application"}
              </ButtonLink>
            </div>
          </Card>
          <Card className="border-gold/25">
            <p className="font-display text-lg text-porcelain">{zh ? "以企业身份询价" : "Inquire as an enterprise"}</p>
            <p className="mt-2 text-sm leading-relaxed text-mist">
              {zh
                ? "看到适合企业礼赠的方向？把它作为线索交给 Maison 礼宾团队：数量、预算、发票、样品路径与分批交付都会在报价前逐项确认。"
                : "Found a direction that fits an enterprise gifting program? Hand it to the Maison concierge team as a lead: quantity, budget, invoicing, sample path, and staged delivery are confirmed item by item before quotation."}
            </p>
            <div className="mt-4">
              <ButtonLink href="/maison#enterprise" variant="outline">
                {zh ? "开启企业线索" : "Open an enterprise lead"}
              </ButtonLink>
            </div>
          </Card>
        </div>
      </Section>
    </>
  );
}
