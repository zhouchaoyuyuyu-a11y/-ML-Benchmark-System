import Link from "next/link";
import type { Metadata } from "next";
import { ButtonLink, Card, Section, SectionHeader, Tag } from "@/components/ui";
import type { CaseStudy } from "@/lib/types";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Case studies — from one sentence to one delivery",
  description:
    "How ZOTAIX turned emotions, relationships, scenarios, and budgets into delivered objects: enterprise gifting programs, co-created bottles, personal bespoke fragrances, and emotional supply runs.",
  path: "/cases",
});

function categorySlug(category: string): string {
  return category.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

function CaseCard({ c, zh }: { c: CaseStudy; zh: boolean }) {
  return (
    <Link href={`/cases/${c.slug}`} className="block h-full">
      <Card hover className="flex h-full flex-col">
        <div className="flex flex-wrap items-center gap-2">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">{c.category}</p>
          {c.featured && <Tag tone="gold">{zh ? "精选" : "Featured"}</Tag>}
        </div>
        <p className="font-display mt-2 text-lg leading-snug text-porcelain">{zh ? c.title_zh : c.title}</p>
        <p className="mt-2 text-sm leading-relaxed text-mist">{c.summary}</p>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {c.emotion_tags.map((t) => (
            <Tag key={t}>{t}</Tag>
          ))}
        </div>
        <p className="mt-4 border-t border-hairline pt-3 text-xs leading-relaxed text-jade">
          <span className="uppercase tracking-wider">{zh ? "成果 · " : "Outcome · "}</span>
          {c.outcome}
        </p>
        <p className="mt-auto pt-3 text-xs text-gold">{zh ? "阅读案例" : "Read the case"} →</p>
      </Card>
    </Link>
  );
}

export default async function CasesPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const cases = [...data.case_studies].sort((a, b) => Number(b.featured) - Number(a.featured));
  const featured = cases.filter((c) => c.featured);
  const categories = Array.from(new Set(data.case_studies.map((c) => c.category)));

  return (
    <>
      <div className="zx-grid-bg border-b border-hairline">
        <Section className="py-14 sm:py-20">
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.25em] text-gold">
            {zh ? "案例" : "Cases"}
          </p>
          <h1 className="font-display max-w-4xl text-3xl leading-tight text-porcelain sm:text-5xl">
            {zh ? "从一句话，到一次交付" : "From one sentence to one delivery"}
          </h1>
          <p className="mt-5 max-w-3xl text-sm leading-relaxed text-mist sm:text-lg">
            {zh
              ? "每个案例都始于同一件事：有人对礼宾说出了一种情绪、一段关系或一个场景。这里记录它们如何成为被交付、被归档的对象。"
              : "Every case begins the same way: someone told the concierge about an emotion, a relationship, or a moment. These are the records of how those sentences became delivered, archived objects."}
          </p>
          {/* Category filter chips */}
          <div className="mt-8 flex flex-wrap gap-2">
            <a
              href="#featured"
              className="inline-flex items-center rounded-full border border-gold/40 px-3 py-1 text-xs text-gold transition-colors hover:bg-gold/10"
            >
              {zh ? "精选" : "Featured"}
            </a>
            {categories.map((cat) => (
              <a
                key={cat}
                href={`#cat-${categorySlug(cat)}`}
                className="inline-flex items-center rounded-full border border-hairline px-3 py-1 text-xs text-mist transition-colors hover:border-gold hover:text-gold"
              >
                {cat}
              </a>
            ))}
          </div>
        </Section>
      </div>

      {/* Featured band */}
      <Section id="featured" className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "精选案例" : "Featured cases"}
          title={zh ? "先读这两个" : "Start with these two"}
        />
        <div className="mt-8 grid gap-5 lg:grid-cols-2">
          {featured.map((c) => (
            <CaseCard key={c.id} c={c} zh={zh} />
          ))}
        </div>
      </Section>

      {/* Grouped by category */}
      <div className="border-t border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "按类别浏览" : "Browse by category"}
            title={zh ? "四种客户，同一条定制链" : "Four kinds of clients, one customization chain"}
            description={
              zh
                ? "企业礼赠、社区共创、私人高定与情绪补给——不同的入口，同样以对象与档案收尾。"
                : "Enterprise gifting, community co-creation, personal bespoke, and emotional supply — different entrances, all ending in an object and an archive."
            }
          />
          <div className="mt-10 space-y-12">
            {categories.map((cat) => {
              const group = cases.filter((c) => c.category === cat);
              return (
                <div key={cat} id={`cat-${categorySlug(cat)}`} className="scroll-mt-24">
                  <div className="flex flex-wrap items-center gap-3">
                    <h3 className="font-display text-xl text-porcelain">{cat}</h3>
                    <Tag tone="gold">
                      {group.length} {zh ? "个案例" : group.length === 1 ? "case" : "cases"}
                    </Tag>
                  </div>
                  <div className="mt-5 grid gap-4 lg:grid-cols-2">
                    {group.map((c) => (
                      <CaseCard key={c.id} c={c} zh={zh} />
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </Section>
      </div>

      {/* CTA */}
      <Section className="py-14 sm:py-20">
        <div className="zx-card flex flex-col items-start gap-4 border-gold/25 p-6 sm:flex-row sm:items-center sm:justify-between sm:p-8">
          <div>
            <p className="font-display text-xl text-porcelain">
              {zh ? "下一个案例，从你的一句话开始" : "The next case starts with your sentence"}
            </p>
            <p className="mt-1.5 text-sm text-mist">
              {zh
                ? "告诉礼宾你的情绪、对象、场景与预算——先创造并保存对象，再决定它成为什么。"
                : "Tell the concierge your emotion, recipient, scenario, and budget — create and save the object first, then decide what it becomes."}
            </p>
          </div>
          <ButtonLink href="/concierge" variant="gold" className="shrink-0">
            {zh ? "启动 AI 礼宾" : "Start the AI concierge"}
          </ButtonLink>
        </div>
      </Section>
    </>
  );
}
