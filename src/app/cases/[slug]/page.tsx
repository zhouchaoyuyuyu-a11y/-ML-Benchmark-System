import Link from "next/link";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { ButtonLink, Card, DefinitionRow, Meridian, Section, Tag } from "@/components/ui";
import { siteUrl } from "@/lib/config";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const c = db().case_studies.find((x) => x.slug === slug);
  if (!c) {
    return pageMetadata({
      title: "Case study",
      description: "A ZOTAIX case study: how an emotion, a relationship, and a budget became a delivered object.",
      path: `/cases/${slug}`,
    });
  }
  return pageMetadata({
    title: c.title,
    description: c.summary,
    path: `/cases/${slug}`,
    image: `${siteUrl}/api/og?title=${encodeURIComponent(c.title)}&subtitle=${encodeURIComponent(c.category)}`,
    keywords: [c.category, c.client_type, ...c.emotion_tags],
  });
}

function relatedFor(category: string, zh: boolean): { href: string; title: string; desc: string } {
  switch (category) {
    case "Co-creation":
      return {
        href: "/co-create",
        title: zh ? "共创铸造池" : "Co-Creation Pool",
        desc: zh
          ? "十人即可开启公开共创页，百瓶解锁风味方向评审。"
          : "Ten people open a public page; a hundred bottles unlock flavor-direction review.",
      };
    case "Enterprise gifting":
      return {
        href: "/maison",
        title: "Maison ZOTAIX",
        desc: zh
          ? "企业礼赠与私人高定：人工礼宾、打样路径与正式报价。"
          : "Enterprise gifting and private bespoke: human concierge, sample paths, and formal quotations.",
      };
    case "Emotional supply":
      return {
        href: "/supply",
        title: "ZOTAIX Supply",
        desc: zh
          ? "轻量、俏皮、可分享的情绪补给对象——含零酒精选项。"
          : "Light, playful, shareable emotional supply objects — zero-proof options included.",
      };
    default:
      return {
        href: "/forge",
        title: zh ? "Forge · AI 编排" : "Forge · AI orchestration",
        desc: zh
          ? "把一种情绪变成酒体方向、香氛方向、命名与瓶身文案。"
          : "Turn one emotion into a liquid direction, fragrance direction, names, and label copy.",
      };
  }
}

export default async function CaseDetailPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const c = data.case_studies.find((x) => x.slug === slug);
  if (!c) notFound();

  const related = relatedFor(c.category, zh);
  const moreCases = data.case_studies.filter((x) => x.slug !== c.slug).slice(0, 3);

  return (
    <>
      {/* Header */}
      <div className="zx-grid-bg border-b border-hairline">
        <Section className="py-12 sm:py-16">
          <Link href="/cases" className="text-xs text-mist transition-colors hover:text-gold">
            ← {zh ? "返回全部案例" : "Back to all cases"}
          </Link>
          <div className="mt-5 flex flex-wrap items-center gap-2">
            <Tag tone="gold">{c.category}</Tag>
            <Tag>{c.client_type}</Tag>
            {c.featured && <Tag tone="jade">{zh ? "精选案例" : "Featured case"}</Tag>}
          </div>
          <h1 className="font-display mt-4 max-w-4xl text-3xl leading-tight text-porcelain sm:text-4xl">
            {zh ? c.title_zh : c.title}
          </h1>
          <p className="mt-4 max-w-3xl text-sm leading-relaxed text-mist sm:text-base">{c.summary}</p>
        </Section>
      </div>

      <Section className="py-12 sm:py-16">
        <div className="grid gap-10 lg:grid-cols-[1fr_340px]">
          {/* Story */}
          <article>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
              {zh ? "案例经过" : "How it unfolded"}
            </p>
            <div className="mt-5 space-y-6">
              {c.story.map((paragraph, i) => (
                <div key={i} className="flex gap-4">
                  <p className="font-display shrink-0 pt-0.5 text-lg text-gold/50">{String(i + 1).padStart(2, "0")}</p>
                  <p className="text-sm leading-loose text-porcelain/90 sm:text-base sm:leading-loose">{paragraph}</p>
                </div>
              ))}
            </div>

            {/* Outcome panel */}
            <div className="zx-card mt-10 border-jade/30 p-6 sm:p-8">
              <p className="text-xs font-semibold uppercase tracking-[0.25em] text-jade">
                {zh ? "成果" : "Outcome"}
              </p>
              <p className="font-display mt-3 text-lg leading-relaxed text-porcelain sm:text-xl">{c.outcome}</p>
              <Meridian className="my-5" />
              <div className="flex flex-wrap gap-2">
                {c.emotion_tags.map((t) => (
                  <Tag key={t} tone="jade">
                    {t}
                  </Tag>
                ))}
              </div>
            </div>
          </article>

          {/* Fact sheet + related */}
          <aside className="space-y-5">
            <Card>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
                {zh ? "案例档案" : "Case record"}
              </p>
              <dl className="mt-3">
                <DefinitionRow label={zh ? "类别" : "Category"}>{c.category}</DefinitionRow>
                <DefinitionRow label={zh ? "客户类型" : "Client type"}>{c.client_type}</DefinitionRow>
                <DefinitionRow label={zh ? "情绪关键词" : "Emotional keywords"}>
                  <span className="flex flex-wrap gap-1.5">
                    {c.emotion_tags.map((t) => (
                      <Tag key={t}>{t}</Tag>
                    ))}
                  </span>
                </DefinitionRow>
                <DefinitionRow label={zh ? "归档日期" : "Archived on"}>{c.created_at.slice(0, 10)}</DefinitionRow>
              </dl>
            </Card>

            <Link href={related.href} className="block">
              <Card hover>
                <p className="text-xs uppercase tracking-wider text-mist">
                  {zh ? "这类定制从这里开始" : "Where this kind of work starts"}
                </p>
                <p className="font-display mt-1.5 text-base text-porcelain">{related.title}</p>
                <p className="mt-1.5 text-sm leading-relaxed text-mist">{related.desc}</p>
                <p className="mt-3 text-xs text-gold">{zh ? "进入" : "Enter"} →</p>
              </Card>
            </Link>

            <Card>
              <p className="text-xs uppercase tracking-wider text-mist">{zh ? "更多案例" : "More cases"}</p>
              <div className="mt-2 space-y-2">
                {moreCases.map((m) => (
                  <Link key={m.id} href={`/cases/${m.slug}`} className="group block border-b border-hairline pb-2 last:border-0 last:pb-0">
                    <p className="text-xs uppercase tracking-wider text-gold/70">{m.category}</p>
                    <p className="font-display mt-0.5 text-sm text-porcelain transition-colors group-hover:text-gold">
                      {zh ? m.title_zh : m.title}
                    </p>
                  </Link>
                ))}
              </div>
            </Card>
          </aside>
        </div>
      </Section>

      {/* CTA band */}
      <div className="border-t border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-16">
          <div className="zx-card flex flex-col items-start gap-4 border-gold/25 p-6 sm:flex-row sm:items-center sm:justify-between sm:p-8">
            <div>
              <p className="font-display text-xl text-porcelain">{zh ? "开始你自己的定制" : "Start your own"}</p>
              <p className="mt-1.5 max-w-xl text-sm text-mist">
                {zh
                  ? "一句话就够了。礼宾会先理解你的情绪、对象、场景与预算，再生成属于你的对象——你保存它，然后决定它成为什么。"
                  : "One sentence is enough. The concierge understands your emotion, recipient, scenario, and budget, then generates your object — you save it, then decide what it becomes."}
              </p>
            </div>
            <div className="flex shrink-0 flex-wrap gap-3">
              <ButtonLink href="/concierge" variant="gold">
                {zh ? "启动 AI 礼宾" : "Start the AI concierge"}
              </ButtonLink>
              <ButtonLink href={related.href} variant="outline">
                {related.title}
              </ButtonLink>
            </div>
          </div>
        </Section>
      </div>
    </>
  );
}
