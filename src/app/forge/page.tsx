import type { Metadata } from "next";
import Link from "next/link";
import AgeGate from "@/components/AgeGate";
import {
  ButtonLink,
  Card,
  EmptyState,
  Notice,
  PageHero,
  Section,
  SectionHeader,
  StatusPill,
  Tag,
} from "@/components/ui";
import { getSessionUser } from "@/lib/auth";
import { complianceNotice } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";
import ForgeClient from "./ForgeClient";

export const metadata: Metadata = pageMetadata({
  title: "Forge — AI orchestration for structured objects",
  description:
    "The ZOTAIX Forge turns object type, emotion, recipient, scenario, budget, and style into structured proposals — liquid directions, fragrance directions, names, and label copy — that flow into Studio, Design, Trade, and Reserve.",
  path: "/forge",
});

export default async function ForgePage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const user = await getSessionUser();
  const data = db();

  const myDrafts = user
    ? data.object_drafts
        .filter((d) => d.user_id === user.id)
        .slice()
        .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
        .slice(0, 4)
    : [];
  const exampleDrafts = user ? [] : data.object_drafts.filter((d) => d.public_visible).slice(0, 2);
  const drafts = user ? myDrafts : exampleDrafts;
  const isExample = !user;

  const chain = [
    {
      step: "01",
      href: "/forge",
      name: "Forge",
      sub: zh ? "AI 编排" : "AI orchestration",
      dEn: "Structured inputs — object type, emotion, recipient, scenario, budget, style — become a concept proposal.",
      dZh: "对象类型、情绪、赠予对象、场景、预算与风格，在这里成为一份结构化提案。",
    },
    {
      step: "02",
      href: "/studio",
      name: "Studio",
      sub: zh ? "视觉预览" : "Visual preview",
      dEn: "Preview the bottle, label, packaging, and emotional card the proposal implies.",
      dZh: "预览提案对应的瓶身、标签、包装与情绪卡片。",
    },
    {
      step: "03",
      href: "/design",
      name: "Design",
      sub: zh ? "提案与版本" : "Proposals & versions",
      dEn: "Saved proposals live here as drafts, with named versions to compare and refine.",
      dZh: "保存的提案在这里成为草案，带可比较、可打磨的命名版本。",
    },
    {
      step: "04",
      href: "/trade",
      name: "Trade",
      sub: zh ? "报价与铸造" : "Quotes & casting",
      dEn: "Physical casting, quotations, and rights — every request passes human review and compliance checks.",
      dZh: "实体铸造、报价与授权——每个请求都经过人工审核与合规检查。",
    },
    {
      step: "05",
      href: "/reserve",
      name: "Reserve",
      sub: zh ? "档案馆" : "The archive",
      dEn: "Archived objects receive a ZOTAIX ID, QR binding, certificate page, and aftercare entry.",
      dZh: "入档对象获得 ZOTAIX ID、QR 绑定、证书页与售后入口。",
    },
  ];

  return (
    <>
      <AgeGate zh={zh} />

      <PageHero
        eyebrow={zh ? "Forge · AI 编排中心" : "Forge · AI Orchestration Center"}
        title={zh ? "结构化输入，在这里成为对象" : "Where structured inputs become objects"}
        description={
          zh
            ? "Forge 是 ZOTAIX 的任务编排中心：选择对象类型，给出情绪、赠予对象、场景、预算与风格偏好，AI 生成结构化提案——酒体方向、香氛方向、命名、瓶身文案与数字印记——并作为业务对象流入 Studio、Design、Trade 与 Reserve。"
            : "The Forge is ZOTAIX's task orchestration center: pick an object type, give it emotion, recipient, scenario, budget, and style preference, and the AI produces a structured proposal — liquid direction, fragrance direction, names, label copy, digital mark — that flows into Studio, Design, Trade, and Reserve as a business object."
        }
      >
        <ButtonLink href="#generator" variant="gold">
          {zh ? "铸造一个对象" : "Forge an object"}
        </ButtonLink>
        <ButtonLink href="/concierge" variant="outline">
          {zh ? "改用对话式礼宾" : "Talk to the concierge instead"}
        </ButtonLink>
        <ButtonLink href="/design" variant="outline">
          {zh ? "查看我的草案" : "My drafts in Design"}
        </ButtonLink>
      </PageHero>

      {/* Forge chain */}
      <Section className="py-12 sm:py-16">
        <SectionHeader
          eyebrow={zh ? "五步链路" : "The five-step chain"}
          title={zh ? "Forge → Studio → Design → Trade → Reserve" : "Forge → Studio → Design → Trade → Reserve"}
          description={
            zh
              ? "每个在 Forge 生成的对象都沿同一条链路移动：从结构化提案，到视觉预览，到版本化草案，到人工审核的报价，最终成为带身份的档案记录。"
              : "Every object generated in the Forge moves along one chain: from structured proposal, to visual preview, to versioned draft, to human-reviewed quotation, and finally to an archived record with an identity."
          }
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {chain.map((c) => (
            <Link key={c.step} href={c.href}>
              <Card hover className="h-full">
                <p className="font-display text-3xl text-gold/60">{c.step}</p>
                <p className="font-display mt-2 text-base text-porcelain">{c.name}</p>
                <p className="text-xs uppercase tracking-wider text-gold">{c.sub}</p>
                <p className="mt-2 text-xs leading-relaxed text-mist">{zh ? c.dZh : c.dEn}</p>
              </Card>
            </Link>
          ))}
        </div>
      </Section>

      {/* Generator */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section id="generator" className="py-12 sm:py-16">
          <SectionHeader
            eyebrow={zh ? "结构化生成器" : "Structured generator"}
            title={zh ? "给 Forge 五个输入，换一份提案" : "Give the Forge five inputs, get one proposal"}
            description={
              zh
                ? "先创造并保存对象——之后再决定它是否成为实体。这里没有购物车，只有提案与去向。"
                : "Create and save the object first — then decide whether it becomes physical. There is no cart here, only proposals and destinations."
            }
          />
          <div className="mt-8">
            <ForgeClient zh={zh} signedIn={!!user} />
          </div>
        </Section>
      </div>

      {/* Recent drafts */}
      <Section className="py-12 sm:py-16">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <SectionHeader
            eyebrow={zh ? "对象草案" : "Object drafts"}
            title={
              user
                ? zh
                  ? "你最近铸造的对象"
                  : "Your recent objects from the Forge"
                : zh
                  ? "在 ZOTAIX 上被铸造的对象"
                  : "Objects forged on ZOTAIX"
            }
            description={
              user
                ? zh
                  ? "保存的提案会出现在这里与 Design 中，每个草案都可以继续演化为版本、档案或铸造请求。"
                  : "Saved proposals appear here and in Design. Every draft can evolve into versions, an archive record, or a casting request."
                : zh
                  ? "两个公开示例，展示 Forge 输出的形态。登录后，这里会显示你自己的草案。"
                  : "Two public examples showing what the Forge produces. Sign in and this section shows your own drafts."
            }
          />
          <ButtonLink href="/design" variant="outline">
            {zh ? "打开 Design" : "Open Design"}
          </ButtonLink>
        </div>

        {drafts.length > 0 ? (
          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            {drafts.map((d) => (
              <Link key={d.id} href="/design">
                <Card hover className="h-full">
                  <div className="flex flex-wrap items-center gap-2">
                    <Tag tone="gold">{d.object_type.replace(/_/g, " ")}</Tag>
                    {isExample && <Tag tone="supply">{zh ? "示例" : "example"}</Tag>}
                    <StatusPill status={d.status} />
                    <span className="ml-auto text-xs text-mist">{d.updated_at.slice(0, 10)}</span>
                  </div>
                  <p className="font-display mt-3 text-lg text-porcelain">{d.title}</p>
                  {(d.scene || d.recipient || d.budget) && (
                    <p className="mt-1 text-xs text-mist">
                      {[d.scene, d.recipient, d.budget].filter(Boolean).join(" · ")}
                    </p>
                  )}
                  {d.label_copy && (
                    <blockquote className="mt-3 border-l-2 border-gold pl-3 font-display text-sm italic text-porcelain">
                      “{d.label_copy}”
                    </blockquote>
                  )}
                  <div className="mt-3 flex flex-wrap gap-2">
                    {d.emotion_tags.map((t) => (
                      <Tag key={t}>{t}</Tag>
                    ))}
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        ) : (
          <div className="mt-8">
            <EmptyState
              title={zh ? "还没有属于你的对象" : "No objects of yours yet"}
              description={
                zh
                  ? "在上方生成器里给出一种情绪与一个对象类型，第一份提案一分钟内就会出现。"
                  : "Give the generator above one emotion and one object type — your first proposal arrives within a minute."
              }
              action={
                <ButtonLink href="#generator" variant="gold">
                  {zh ? "铸造第一个对象" : "Forge your first object"}
                </ButtonLink>
              }
            />
          </div>
        )}

        <div className="mt-10">
          <Notice tone="gold">{zh ? complianceNotice.zh : complianceNotice.en}</Notice>
        </div>
      </Section>
    </>
  );
}
