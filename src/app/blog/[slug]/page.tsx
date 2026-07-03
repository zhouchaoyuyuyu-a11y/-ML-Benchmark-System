import Link from "next/link";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { Fragment } from "react";
import { ButtonLink, Card, Meridian, Section, Tag } from "@/components/ui";
import { siteUrl } from "@/lib/config";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const { slug } = await params;
  const post = db().blog_posts.find((p) => p.slug === slug);
  if (!post) {
    return pageMetadata({
      title: "Journal entry",
      description: "An essay from the ZOTAIX journal — notes from the team building the AI concierge customization platform.",
      path: `/blog/${slug}`,
    });
  }
  return pageMetadata({
    title: post.title,
    description: post.excerpt,
    path: `/blog/${slug}`,
    image: `${siteUrl}/api/og?title=${encodeURIComponent(post.title)}&subtitle=${encodeURIComponent(post.category)}`,
    keywords: [post.category, "ZOTAIX journal"],
  });
}

function formatDate(iso: string, zh: boolean): string {
  return new Intl.DateTimeFormat(zh ? "zh-CN" : "en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  }).format(new Date(iso));
}

/** Split a paragraph into its first sentence and the remainder. */
function splitFirstSentence(paragraph: string): { quote: string; rest: string } | null {
  const match = paragraph.match(/^[^.。!?！？]+[.。!?！？]["'”’]?/);
  if (!match) return null;
  const quote = match[0].trim();
  const rest = paragraph.slice(match[0].length).trim();
  if (!rest) return null;
  return { quote, rest };
}

export default async function BlogArticlePage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const post = data.blog_posts.find((p) => p.slug === slug);
  if (!post) notFound();

  const related = data.blog_posts
    .filter((p) => p.slug !== post.slug && p.category === post.category)
    .slice(0, 3);
  const fallbackRelated = data.blog_posts.filter((p) => p.slug !== post.slug).slice(0, 3);
  const relatedPosts = related.length > 0 ? related : fallbackRelated;

  const pullQuoteIndex = post.body.length >= 3 ? 2 : -1;

  return (
    <>
      {/* Article header */}
      <div className="zx-grid-bg border-b border-hairline">
        <Section className="py-12 sm:py-16">
          <Link href="/blog" className="text-xs text-mist transition-colors hover:text-gold">
            ← {zh ? "返回品牌志" : "Back to the journal"}
          </Link>
          <div className="mt-5 flex flex-wrap items-center gap-2">
            <Tag tone="gold">{post.category}</Tag>
            {post.featured && <Tag>{zh ? "精选" : "Featured"}</Tag>}
          </div>
          <h1 className="font-display mt-4 max-w-4xl text-3xl leading-tight text-porcelain sm:text-4xl">
            {zh ? post.title_zh : post.title}
          </h1>
          <p className="mt-4 max-w-3xl text-sm leading-relaxed text-mist sm:text-base">{post.excerpt}</p>
          <div className="mt-6 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-mist">
            <span className="text-porcelain">{post.author}</span>
            <span aria-hidden="true">·</span>
            <time dateTime={post.published_at}>{formatDate(post.published_at, zh)}</time>
          </div>
        </Section>
      </div>

      {/* Body */}
      <Section className="py-12 sm:py-16">
        <article className="mx-auto max-w-2xl">
          <div className="space-y-7">
            {post.body.map((paragraph, i) => {
              if (i === pullQuoteIndex) {
                const split = splitFirstSentence(paragraph);
                if (split) {
                  return (
                    <Fragment key={i}>
                      <blockquote className="border-l-2 border-gold/60 py-1 pl-5">
                        <p className="font-display text-xl leading-relaxed text-gold sm:text-2xl">{split.quote}</p>
                      </blockquote>
                      <p className="text-sm leading-loose text-porcelain/90 sm:text-base sm:leading-loose">
                        {split.rest}
                      </p>
                    </Fragment>
                  );
                }
              }
              return (
                <p key={i} className="text-sm leading-loose text-porcelain/90 sm:text-base sm:leading-loose">
                  {paragraph}
                </p>
              );
            })}
          </div>

          <Meridian className="my-10" />
          <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-mist">
            <p>
              {zh ? "撰文 · " : "Written by "}
              <span className="text-porcelain">{post.author}</span>
            </p>
            <p>
              {zh ? "发布于 " : "Published "}
              <time dateTime={post.published_at}>{formatDate(post.published_at, zh)}</time>
            </p>
          </div>
        </article>
      </Section>

      {/* Related posts */}
      <div className="border-t border-hairline bg-obsidian/40">
        <Section className="py-12 sm:py-16">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
                {zh ? "延伸阅读" : "Keep reading"}
              </p>
              <h2 className="font-display mt-2 text-2xl text-porcelain">
                {related.length > 0
                  ? zh
                    ? `更多「${post.category}」文章`
                    : `More from ${post.category}`
                  : zh
                    ? "来自品牌志的更多文章"
                    : "More from the journal"}
              </h2>
            </div>
            <ButtonLink href="/blog" variant="outline">
              {zh ? "全部文章" : "All entries"}
            </ButtonLink>
          </div>
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {relatedPosts.map((p) => (
              <Link key={p.id} href={`/blog/${p.slug}`} className="block h-full">
                <Card hover className="flex h-full flex-col">
                  <div className="flex flex-wrap items-center gap-2">
                    <Tag>{p.category}</Tag>
                    <span className="text-xs text-mist">{formatDate(p.published_at, zh)}</span>
                  </div>
                  <p className="font-display mt-2.5 text-base leading-snug text-porcelain">
                    {zh ? p.title_zh : p.title}
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-mist">{p.excerpt}</p>
                  <p className="mt-auto pt-3 text-xs text-gold">{zh ? "阅读" : "Read"} →</p>
                </Card>
              </Link>
            ))}
          </div>
        </Section>
      </div>

      {/* CTA band */}
      <Section className="py-14 sm:py-16">
        <div className="zx-card flex flex-col items-start gap-4 border-gold/25 p-6 sm:flex-row sm:items-center sm:justify-between sm:p-8">
          <div>
            <p className="font-display text-xl text-porcelain">
              {zh ? "把想法变成一个对象" : "Turn the idea into an object"}
            </p>
            <p className="mt-1.5 max-w-xl text-sm text-mist">
              {zh
                ? "礼宾会先理解你的情绪、对象、场景与预算，再生成可保存、可归档的对象——之后再决定它是否成为实物。"
                : "The concierge understands your emotion, recipient, scenario, and budget first, then generates an object you can save and archive — whether it becomes physical remains entirely your choice."}
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
