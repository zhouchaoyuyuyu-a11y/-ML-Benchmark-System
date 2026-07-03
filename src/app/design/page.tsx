import type { Metadata } from "next";
import {
  ButtonLink,
  Card,
  DefinitionRow,
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
import type { DesignVersion, ObjectDraft } from "@/lib/types";
import ArchiveButton from "./ArchiveButton";

export const metadata: Metadata = pageMetadata({
  title: "Design — structured proposals and versioned directions",
  description:
    "The ZOTAIX Design module stores names, label copy, fragrance directions, liquid directions, visual styles, and packaging directions as versioned proposals — each version carrying a citable, tamper-evident hash.",
  path: "/design",
});

function DraftCard({
  draft,
  versions,
  zh,
  own,
  example,
}: {
  draft: ObjectDraft;
  versions: DesignVersion[];
  zh: boolean;
  own: boolean;
  example: boolean;
}) {
  return (
    <Card className="h-full">
      <div className="flex flex-wrap items-center gap-2">
        <Tag tone="gold">{draft.object_type.replace(/_/g, " ")}</Tag>
        {example && <Tag tone="supply">{zh ? "公开示例" : "public example"}</Tag>}
        <StatusPill status={draft.status} />
        <span className="ml-auto text-xs text-mist">{draft.updated_at.slice(0, 10)}</span>
      </div>

      <p className="font-display mt-3 text-xl text-porcelain">{draft.title}</p>

      {draft.emotion_tags.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {draft.emotion_tags.map((t) => (
            <Tag key={t}>{t}</Tag>
          ))}
        </div>
      )}

      <dl className="mt-4">
        {(draft.scene || draft.recipient || draft.budget) && (
          <DefinitionRow label={zh ? "场景 · 对象 · 预算" : "Scene · Recipient · Budget"}>
            {[draft.scene, draft.recipient, draft.budget].filter(Boolean).join(" · ")}
          </DefinitionRow>
        )}
        {draft.names && draft.names.length > 0 && (
          <DefinitionRow label={zh ? "候选命名" : "Names"}>{draft.names.join(" / ")}</DefinitionRow>
        )}
        {draft.label_copy && (
          <DefinitionRow label={zh ? "瓶身文案" : "Label copy"}>
            <span className="font-display italic">“{draft.label_copy}”</span>
          </DefinitionRow>
        )}
        {draft.liquid_direction && (
          <DefinitionRow label={zh ? "酒体方向" : "Liquid direction"}>{draft.liquid_direction}</DefinitionRow>
        )}
        {draft.scent_direction && (
          <DefinitionRow label={zh ? "香氛方向" : "Fragrance direction"}>{draft.scent_direction}</DefinitionRow>
        )}
        {draft.visual_style && (
          <DefinitionRow label={zh ? "视觉风格" : "Visual style"}>{draft.visual_style}</DefinitionRow>
        )}
      </dl>

      <div className="mt-4">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
          {zh ? `版本 · ${versions.length}` : `Versions · ${versions.length}`}
        </p>
        {versions.length > 0 ? (
          <div className="mt-2 space-y-3">
            {versions.map((v) => (
              <div key={v.id} className="rounded-lg border border-hairline bg-ink/40 p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="font-display text-sm text-porcelain">{v.version_name}</p>
                  <code className="rounded bg-veil px-2 py-0.5 font-mono text-xs text-gold">{v.version_hash}</code>
                  <span className="ml-auto text-xs text-mist">{v.created_at.slice(0, 10)}</span>
                </div>
                <div className="mt-3 grid gap-1.5 sm:grid-cols-2">
                  {Object.entries(v.design_payload).map(([key, value]) => (
                    <p key={key} className="text-xs leading-relaxed text-mist">
                      <span className="uppercase tracking-wider text-porcelain/70">{key}</span> · {value}
                    </p>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-2 text-xs text-mist">
            {zh
              ? "此草案暂无独立版本——从 Forge 或 Studio 保存的每一次打磨，都会以带哈希的快照形式出现在这里。"
              : "No standalone versions on this draft yet — every refinement saved from the Forge or Studio arrives here as a hashed snapshot."}
          </p>
        )}
      </div>

      <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-hairline pt-4">
        <ButtonLink href="/forge" variant="outline">
          {zh ? "在 Forge 继续" : "Continue in Forge"}
        </ButtonLink>
        <ButtonLink href="/studio" variant="outline">
          {zh ? "在 Studio 预览" : "Preview in Studio"}
        </ButtonLink>
        {own && <ArchiveButton draftId={draft.id} zh={zh} />}
      </div>
    </Card>
  );
}

export default async function DesignPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const user = await getSessionUser();
  const data = db();

  const versionsFor = (draftId: string) =>
    data.design_versions
      .filter((v) => v.object_draft_id === draftId)
      .slice()
      .sort((a, b) => a.created_at.localeCompare(b.created_at));

  const myDrafts = user
    ? data.object_drafts
        .filter((d) => d.user_id === user.id)
        .slice()
        .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
    : [];
  const publicDrafts = user ? [] : data.object_drafts.filter((d) => d.public_visible);

  const stored = [
    { en: "Names", zh: "候选命名", dEn: "Three to five naming candidates per object, kept side by side.", dZh: "每个对象保留三到五个命名候选，并列存放。" },
    { en: "Label copy", zh: "瓶身文案", dEn: "The single line that carries the emotion onto glass, card, and certificate.", dZh: "把情绪带上玻璃、卡片与证书的那一句话。" },
    { en: "Fragrance direction", zh: "香氛方向", dEn: "Top, heart, and base notes as a direction the atelier can compose from.", dZh: "以前、中、尾调描述的方向，供调香工作室执行。" },
    { en: "Liquid direction", zh: "酒体方向", dEn: "Base liquid, infusion, and ABV expressed as an honest, reviewable brief.", dZh: "基酒、浸渍与度数——一份诚实、可审核的简报。" },
    { en: "Visual style", zh: "视觉风格", dEn: "Glass tone, silhouette, accent metal, and typography in one sentence.", dZh: "玻璃色调、轮廓、金属点缀与排印风格，一句话说清。" },
    { en: "Label text & layout", zh: "标签文字与排布", dEn: "Vertical serif, horizontal band, or sticker — how the copy sits on the bottle.", dZh: "竖排衬线、横向腰封或贴纸——文案在瓶身上的位置。" },
    { en: "Packaging direction", zh: "包装方向", dEn: "Slipcase, magnetic box, or kraft sleeve, with materials and closures.", dZh: "函套、磁吸盒或牛皮纸套，连同材质与开合方式。" },
  ];

  const lifecycle = [
    {
      step: "01",
      en: "Save creates v1",
      zh: "保存即生成 v1",
      dEn: "Saving a proposal from the Forge, Studio, or Concierge automatically mints a v1 snapshot with its own hash.",
      dZh: "从 Forge、Studio 或礼宾保存提案，会自动生成带独立哈希的 v1 快照。",
    },
    {
      step: "02",
      en: "Refine into named versions",
      zh: "打磨为命名版本",
      dEn: "Each refinement — a new glass tone, a softer ABV, a different band — becomes a named, comparable version.",
      dZh: "每一次打磨——新的玻璃色、更柔和的度数、不同的腰封——都成为可比较的命名版本。",
    },
    {
      step: "03",
      en: "Cite the hash downstream",
      zh: "在下游引用哈希",
      dEn: "Reserve certificates and Trade quotations reference the exact version hash, so what was archived or quoted is never ambiguous.",
      dZh: "档案证书与 Trade 报价单引用确切的版本哈希，被归档或被报价的内容永不含糊。",
    },
  ];

  return (
    <>
      <PageHero
        eyebrow={zh ? "Design · 提案与版本系统" : "Design · Proposals & Version System"}
        title={zh ? "每个方向都值得一个可引用的版本" : "Every direction deserves a citable version"}
        description={
          zh
            ? "Design 是 ZOTAIX 的结构化提案库：命名、文案、香氛方向、酒体方向、视觉风格、标签文字与包装方向，以版本化提案的形式保存，每个版本携带自己的版本哈希。"
            : "Design is ZOTAIX's structured proposal store: names, copy, fragrance directions, liquid directions, visual styles, label text, and packaging directions live here as versioned proposals — and every version carries its own version hash."
        }
      >
        <ButtonLink href="/forge" variant="gold">
          {zh ? "去 Forge 生成提案" : "Generate a proposal in Forge"}
        </ButtonLink>
        <ButtonLink href="/studio" variant="outline">
          {zh ? "去 Studio 预览视觉" : "Preview visuals in Studio"}
        </ButtonLink>
        <ButtonLink href="/reserve" variant="outline">
          {zh ? "查看档案馆" : "Visit Reserve"}
        </ButtonLink>
      </PageHero>

      {/* What Design stores */}
      <Section className="py-12 sm:py-16">
        <SectionHeader
          eyebrow={zh ? "Design 保存什么" : "What Design stores"}
          title={zh ? "七个可版本化的方向" : "Seven directions, all versionable"}
          description={
            zh
              ? "一个对象不是一张图片，而是一组可执行的方向。Design 把它们拆开保存，让每一项都能被单独打磨、比较与引用。"
              : "An object is not one image — it is a set of executable directions. Design stores them separately so each can be refined, compared, and cited on its own."
          }
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {stored.map((s) => (
            <Card key={s.en} className="h-full">
              <p className="font-display text-base text-porcelain">{zh ? s.zh : s.en}</p>
              <p className="mt-1.5 text-sm leading-relaxed text-mist">{zh ? s.dZh : s.dEn}</p>
            </Card>
          ))}
        </div>
        <div className="mt-8">
          <Notice tone="gold" title={zh ? "版本哈希为什么重要" : "Why version hashes matter"}>
            {zh
              ? "每个版本都会获得一个由内容派生的哈希，例如 zx-8f31c2a0。哈希让每个方向都可被引用、且防篡改：Reserve 证书与 Trade 报价单引用的是这个哈希——内容一旦改变，哈希随之改变，被归档与被报价的版本因此永远清晰可查。"
              : "Every version receives a content-derived hash like zx-8f31c2a0. The hash makes each direction citable and tamper-evident: Reserve certificates and Trade quotations reference it, and since any change to the content changes the hash, the exact version that was archived or quoted stays verifiable forever."}
          </Notice>
        </div>
      </Section>

      {/* Lifecycle */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-12 sm:py-16">
          <SectionHeader
            eyebrow={zh ? "版本如何累积" : "How versions accrue"}
            title={zh ? "从第一次保存到最后一次引用" : "From the first save to the last citation"}
          />
          <div className="mt-10 grid gap-4 lg:grid-cols-3">
            {lifecycle.map((l) => (
              <Card key={l.step} className="h-full">
                <p className="font-display text-3xl text-gold/60">{l.step}</p>
                <p className="font-display mt-2 text-base text-porcelain">{zh ? l.zh : l.en}</p>
                <p className="mt-2 text-sm leading-relaxed text-mist">{zh ? l.dZh : l.dEn}</p>
              </Card>
            ))}
          </div>
        </Section>
      </div>

      {/* Drafts */}
      <Section className="py-12 sm:py-16">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <SectionHeader
            eyebrow={zh ? "对象草案" : "Object drafts"}
            title={
              user
                ? zh
                  ? "你的草案与版本"
                  : "Your drafts and their versions"
                : zh
                  ? "公开的草案与版本"
                  : "Public drafts and their versions"
            }
            description={
              user
                ? zh
                  ? "每张卡片是一个对象草案：状态、情绪标签、各项方向，以及它累积的全部版本。可以继续打磨、去 Studio 预览，或直接存入档案馆。"
                  : "Each card is one object draft: status, emotion tags, its directions, and every version it has accrued. Keep refining, preview it in Studio, or archive it to Reserve."
                : zh
                  ? "这些是创作者选择公开的草案，展示 Design 如何保存方向与版本。登录后，这里显示你自己的草案。"
                  : "These are drafts their creators chose to make public, showing how Design stores directions and versions. Sign in and this section shows your own."
            }
          />
          <ButtonLink href="/forge" variant="outline">
            {zh ? "新建一个对象" : "Start a new object"}
          </ButtonLink>
        </div>

        {user ? (
          myDrafts.length > 0 ? (
            <div className="mt-8 grid gap-5 lg:grid-cols-2">
              {myDrafts.map((d) => (
                <DraftCard key={d.id} draft={d} versions={versionsFor(d.id)} zh={zh} own example={false} />
              ))}
            </div>
          ) : (
            <div className="mt-8">
              <EmptyState
                title={zh ? "你的提案库还是空的" : "Your proposal store is empty"}
                description={
                  zh
                    ? "去 Forge 用一种情绪换一份结构化提案，或在 Studio 里搭一只瓶子——保存后它们都会出现在这里，并自动携带 v1 版本。"
                    : "Trade one emotion for a structured proposal in the Forge, or build a bottle in the Studio — once saved, both appear here with an automatic v1 version."
                }
                action={
                  <div className="flex flex-wrap justify-center gap-3">
                    <ButtonLink href="/forge" variant="gold">
                      {zh ? "打开 Forge" : "Open the Forge"}
                    </ButtonLink>
                    <ButtonLink href="/studio" variant="outline">
                      {zh ? "打开 Studio" : "Open the Studio"}
                    </ButtonLink>
                  </div>
                }
              />
            </div>
          )
        ) : (
          <>
            <div className="mt-8 grid gap-5 lg:grid-cols-2">
              {publicDrafts.map((d) => (
                <DraftCard key={d.id} draft={d} versions={versionsFor(d.id)} zh={zh} own={false} example />
              ))}
            </div>
            <div className="mt-8">
              <EmptyState
                title={zh ? "你的草案会出现在这里" : "Your own drafts would live here"}
                description={
                  zh
                    ? "登录后，Design 会显示你保存的全部对象草案、它们的方向与版本历史，并开放「存入档案馆」入口。"
                    : "Sign in and Design lists every object draft you have saved — its directions, its version history, and the entry to archive it in Reserve."
                }
                action={
                  <div className="flex flex-wrap justify-center gap-3">
                    <ButtonLink href="/register" variant="gold">
                      {zh ? "免费注册" : "Register free"}
                    </ButtonLink>
                    <ButtonLink href="/login" variant="outline">
                      {zh ? "登录" : "Sign in"}
                    </ButtonLink>
                  </div>
                }
              />
            </div>
          </>
        )}

        <div className="mt-10">
          <Notice tone="gold">{zh ? complianceNotice.zh : complianceNotice.en}</Notice>
        </div>
      </Section>
    </>
  );
}
