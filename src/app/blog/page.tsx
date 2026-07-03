import Link from "next/link";
import type { Metadata } from "next";
import { Card, Section, SectionHeader, Tag } from "@/components/ui";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Journal — notes from the ZOTAIX atelier",
  description:
    "Essays on objects before carts, small-batch honesty, archives that outlast bottles, and how ZOTAIX treats self-expression tags — written by the team building the platform.",
  path: "/blog",
});

function formatDate(iso: string, zh: boolean): string {
  return new Intl.DateTimeFormat(zh ? "zh-CN" : "en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  }).format(new Date(iso));
}

export default async function BlogPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const posts = [...data.blog_posts].sort(
    (a, b) => new Date(b.published_at).getTime() - new Date(a.published_at).getTime()
  );
  const featured = posts.filter((p) => p.featured);
  const rest = posts.filter((p) => !p.featured);

  return (
    <>
      <div className="zx-grid-bg border-b border-hairline">
        <Section className="py-14 sm:py-20">
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.25em] text-gold">
            {zh ? "品牌志" : "Journal"}
          </p>
          <h1 className="font-display max-w-4xl text-3xl leading-tight text-porcelain sm:text-5xl">
            {zh ? "来自工坊的笔记" : "Notes from the atelier"}
          </h1>
          <p className="mt-5 max-w-3xl text-sm leading-relaxed text-mist sm:text-lg">
            {zh
              ? "关于对象为何先于购物车、小批量定制的诚实边界、比瓶子活得更久的档案，以及平台如何对待自我表达标签——由构建这个平台的人写下。"
              : "On why objects come before carts, the honest boundaries of small-batch customization, archives that outlast bottles, and how the platform treats self-expression tags — written by the people building it."}
          </p>
        </Section>
      </div>

      {/* Featured band */}
      <Section className="py-14 sm:py-20">
        <SectionHeader eyebrow={zh ? "精选" : "Featured"} title={zh ? "先读这些" : "Start here"} />
        <div className="mt-8 grid gap-5 lg:grid-cols-2">
          {featured.map((p) => (
            <Link key={p.id} href={`/blog/${p.slug}`} className="block h-full">
              <Card hover className="flex h-full flex-col">
                <div className="flex flex-wrap items-center gap-2">
                  <Tag tone="gold">{p.category}</Tag>
                  <span className="text-xs text-mist">{formatDate(p.published_at, zh)}</span>
                </div>
                <p className="font-display mt-3 text-xl leading-snug text-porcelain">
                  {zh ? p.title_zh : p.title}
                </p>
                <p className="mt-2.5 text-sm leading-relaxed text-mist">{p.excerpt}</p>
                <div className="mt-auto flex items-center justify-between pt-4">
                  <span className="text-xs text-mist">{p.author}</span>
                  <span className="text-xs text-gold">{zh ? "阅读" : "Read"} →</span>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      </Section>

      {/* All entries */}
      <div className="border-t border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "全部文章" : "All entries"}
            title={zh ? "按时间倒序" : "Latest first"}
          />
          <div className="mt-8 space-y-4">
            {(rest.length > 0 ? rest : posts).map((p) => (
              <Link key={p.id} href={`/blog/${p.slug}`} className="block">
                <Card hover className="!p-5">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <Tag>{p.category}</Tag>
                        <span className="text-xs text-mist">{formatDate(p.published_at, zh)}</span>
                      </div>
                      <p className="font-display mt-2 text-lg leading-snug text-porcelain">
                        {zh ? p.title_zh : p.title}
                      </p>
                      <p className="mt-1.5 text-sm leading-relaxed text-mist">{p.excerpt}</p>
                      <p className="mt-2 text-xs text-mist">{p.author}</p>
                    </div>
                    <span className="shrink-0 text-gold sm:pt-1">→</span>
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        </Section>
      </div>

      {/* Cross-link strip */}
      <Section className="py-12 sm:py-16">
        <div className="grid gap-4 sm:grid-cols-3">
          <Link href="/cases" className="group block">
            <Card hover className="h-full">
              <p className="font-display text-base text-porcelain transition-colors group-hover:text-gold">
                {zh ? "案例" : "Cases"}
              </p>
              <p className="mt-1.5 text-sm text-mist">
                {zh ? "这些想法在真实交付中的样子。" : "What these ideas look like in real deliveries."}
              </p>
            </Card>
          </Link>
          <Link href="/about" className="group block">
            <Card hover className="h-full">
              <p className="font-display text-base text-porcelain transition-colors group-hover:text-gold">
                {zh ? "关于卓序" : "About ZOTAIX"}
              </p>
              <p className="mt-1.5 text-sm text-mist">
                {zh ? "平台定义、五模块对象链与五条原则。" : "The platform definition, the five-module chain, the five principles."}
              </p>
            </Card>
          </Link>
          <Link href="/concierge" className="group block">
            <Card hover className="h-full">
              <p className="font-display text-base text-porcelain transition-colors group-hover:text-gold">
                {zh ? "AI 礼宾" : "The AI concierge"}
              </p>
              <p className="mt-1.5 text-sm text-mist">
                {zh ? "读完之后，试着创造一个属于你的对象。" : "After reading, try creating an object of your own."}
              </p>
            </Card>
          </Link>
        </div>
      </Section>
    </>
  );
}
