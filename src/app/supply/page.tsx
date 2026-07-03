import Link from "next/link";
import type { Metadata } from "next";
import { ButtonLink, Card, Notice, PageHero, ProgressBar, Section, SectionHeader, Tag } from "@/components/ui";
import SupplyClient from "./SupplyClient";
import { complianceNotice, pick } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "ZOTAIX Supply — emotional spirits, fragrances, and shareable cards for everyday moments",
  description:
    "The playful supply line: turn one sentence of feeling into a low-ABV or zero-proof spirit direction, a fragrance direction, a custom label, and a shareable emotional card — starting under 100 RMB.",
  path: "/supply",
  keywords: ["emotional supply", "zero-proof", "low-ABV", "emotional card", "custom label", "ZOTAIX Supply"],
});

export default async function SupplyPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const s = data.settings;
  const projects = data.co_creation_projects
    .filter((p) => p.public_visible && p.review_status === "approved")
    .slice(0, 2);
  const labelExample = data.object_drafts.find((d) => d.id === "dft_003" && d.public_visible);

  const cardParams = new URLSearchParams({
    copy: labelExample?.label_copy ?? "One more page. Then the whole sky.",
    mark: "Night Study Ration · Digital Mark",
    keywords: (labelExample?.emotion_tags ?? ["persistence", "humor", "exam"]).join(" · "),
  });
  const sampleCardUrl = `/api/card?${cardParams.toString()}`;

  const proofLevels = [
    {
      icon: "◌",
      title: zh ? "零酒精 · 0.0%" : "Zero-proof · 0.0%",
      desc: zh
        ? "完全不含酒精。考试季与所有面向未成年人的场景固定使用零酒精配方——气泡、风味与仪式感一样不少。"
        : "Contains no alcohol at all. Exam-season and every minor-safe scenario always use the zero-proof line — same sparkle, flavor, and ritual, no alcohol.",
      tag: zh ? "考试季默认" : "Exam-season default",
    },
    {
      icon: "◐",
      title: zh ? "微醺 · ≤6%" : "Light · ≤6% ABV",
      desc: zh
        ? "气泡果感的轻酒精方向，适合生日聚会与庆祝时刻。实体交付前经年龄与地区合规审核。"
        : "Sparkling, fruit-forward, gently alcoholic — built for birthdays and small celebrations. Age and region checks apply before any physical delivery.",
      tag: zh ? "庆祝场景" : "Celebrations",
    },
    {
      icon: "◑",
      title: zh ? "补给标准 · ≤12%" : "Supply classic · ≤12% ABV",
      desc: zh
        ? "补给线的度数上限。更高度数与陈年液体属于 Maison 高定线，由人工礼宾确认。"
        : "The ceiling for the supply line. Higher-proof and aged liquids belong to the Maison premium line, confirmed by a human concierge.",
      tag: zh ? "补给上限" : "Line ceiling",
    },
  ];

  const labelKnobs = [
    { en: "A name of its own — the AI proposes three, you keep one", zh: "一个专属命名——AI 提三个，你留一个" },
    { en: "One line of label copy that sounds like you", zh: "一句像你说出来的瓶身文案" },
    { en: "Palette and sticker-energy visual style", zh: "配色与贴纸感的视觉风格" },
    { en: "A digital mark and serial for your archive", zh: "写入档案的数字印记与编号" },
  ];

  return (
    <>
      <PageHero
        tone="supply"
        eyebrow={zh ? "ZOTAIX Supply · 情绪补给线" : "ZOTAIX Supply · the emotional supply line"}
        title={
          zh
            ? "一句话的情绪，也值得一瓶正式的补给"
            : "One sentence of feeling deserves a properly bottled supply"
        }
        description={
          zh
            ? "生日、失恋、考试、职场、友情、恋爱——把此刻写成一句话，AI 生成酒体或香氛方向、命名、瓶身文案与可分享的情绪卡片。先创造并保存，再决定要不要让它成真。"
            : "Birthday, breakup, exam, workplace, friendship, romance — write the moment in one sentence and the AI returns a spirit or fragrance direction, names, label copy, and a shareable emotional card. Create and save first; decide later whether it becomes physical."
        }
      >
        <ButtonLink href="#generator" variant="supply">
          {zh ? "开始生成补给" : "Generate my supply"}
        </ButtonLink>
        <ButtonLink href="/co-create" variant="outline">
          {zh ? "进入共创池" : "Enter co-creation"}
        </ButtonLink>
        <ButtonLink href="/membership" variant="outline">
          {zh ? "核心序列" : "Core Sequence"}
        </ButtonLink>
      </PageHero>

      {/* Generator */}
      <Section id="generator" className="scroll-mt-24 py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "补给生成器" : "The generator"}
          title={zh ? "选一个场景，说一句感受" : "Pick a scenario, say one sentence"}
          description={
            zh
              ? "六个场景模板会替你预填情绪与场景；也可以全部手写。切换“情绪之酒 / 香氛补给”，选好预算与度数，交给礼宾。"
              : "Six scenario templates prefill the emotion and scenario for you — or write everything yourself. Toggle between emotional spirit and fragrance supply, set a budget and proof level, and hand it to the concierge."
          }
        />
        <div className="mt-10">
          <SupplyClient zh={zh} />
        </div>
      </Section>

      {/* Proof levels */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "度数说明" : "Proof, explained"}
            title={zh ? "低度数是默认，零酒精是承诺" : "Low proof is the default; zero-proof is a promise"}
            description={
              zh
                ? "补给线为年轻的日常情绪而生，所以酒精是可选项而不是前提。零酒精产品完全不含酒精，可安心用于考试季与未成年人相关场景。"
                : "The supply line exists for young, everyday feelings — so alcohol is an option, never a premise. Zero-proof products contain no alcohol whatsoever, making them safe for exam season and any minor-safe scenario."
            }
          />
          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {proofLevels.map((p) => (
              <Card key={p.title} className="h-full">
                <div className="flex items-center justify-between">
                  <span className="text-lg text-supply">{p.icon}</span>
                  <Tag tone="supply">{p.tag}</Tag>
                </div>
                <p className="font-display mt-3 text-base text-porcelain">{p.title}</p>
                <p className="mt-2 text-sm leading-relaxed text-mist">{p.desc}</p>
              </Card>
            ))}
          </div>
          <div className="mt-8">
            <Notice tone="gold">{pick(locale, complianceNotice)}</Notice>
          </div>
        </Section>
      </div>

      {/* Label customization + emotional card */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "标签与卡片" : "Labels & cards"}
          title={zh ? "你的句子，印在标签上，走进群聊里" : "Your sentence, printed on a label, sent into the group chat"}
          description={
            zh
              ? "每个补给对象都自带可定制的标签系统与一张可下载、可转发的情绪卡片——先分享感受，再考虑瓶子。"
              : "Every supply object ships with a customizable label system and a downloadable, forwardable emotional card — share the feeling first, think about the bottle later."
          }
        />
        <div className="mt-10 grid gap-5 lg:grid-cols-2">
          <Card className="h-full">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-supply">
              {zh ? "标签定制" : "Label customization"}
            </p>
            {labelExample ? (
              <div className="mt-4">
                <div className="flex flex-wrap gap-2">
                  {labelExample.emotion_tags.map((t) => (
                    <Tag key={t} tone="supply">
                      {t}
                    </Tag>
                  ))}
                </div>
                <p className="font-display mt-3 text-lg text-porcelain">{labelExample.title}</p>
                {labelExample.label_copy && (
                  <blockquote className="mt-3 border-l-2 border-supply pl-3 font-display text-base italic text-porcelain">
                    “{labelExample.label_copy}”
                  </blockquote>
                )}
                {labelExample.visual_style && (
                  <p className="mt-3 text-sm leading-relaxed text-mist">
                    {zh ? "视觉风格：" : "Visual style: "}
                    {labelExample.visual_style}
                  </p>
                )}
              </div>
            ) : (
              <p className="mt-4 text-sm leading-relaxed text-mist">
                {zh
                  ? "从共创池与档案馆里可以看到公开的标签范例。"
                  : "Public label examples live in the co-creation pool and the Reserve archive."}
              </p>
            )}
            <ul className="mt-5 space-y-2">
              {labelKnobs.map((k) => (
                <li key={k.en} className="flex items-start gap-2 text-sm text-mist">
                  <span className="mt-0.5 text-supply">✶</span>
                  <span>{zh ? k.zh : k.en}</span>
                </li>
              ))}
            </ul>
            <div className="mt-5">
              <ButtonLink href="/studio" variant="outline">
                {zh ? "打开瓶身与标签预览" : "Open bottle & label preview"}
              </ButtonLink>
            </div>
          </Card>
          <Card className="h-full">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-supply">
              {zh ? "可分享的情绪卡片" : "The shareable emotional card"}
            </p>
            <p className="mt-3 text-sm leading-relaxed text-mist">
              {zh
                ? "每次生成后，提案卡上的“生成情绪卡片”会把你的句子、关键词与数字印记排成一张图——发进群聊、贴进日记、配进朋友圈都刚好。"
                : "After every generation, the proposal's emotional-card button lays your sentence, keywords, and digital mark into an image — sized for group chats, journals, and story posts."}
            </p>
            <div className="mt-4">
              {/* Server-rendered sample card from the platform card API */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={sampleCardUrl}
                alt={zh ? "情绪卡片示例" : "Sample emotional card"}
                className="w-full max-w-md rounded-lg border border-hairline"
              />
              <p className="mt-2 text-xs text-mist">
                {zh
                  ? "示例：来自公开草案《Night Study Ration》的情绪卡片。"
                  : "Sample: an emotional card from the public draft “Night Study Ration”."}
              </p>
            </div>
          </Card>
        </div>
      </Section>

      {/* Co-creation entry */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <SectionHeader
              eyebrow={zh ? "共创入口" : "Co-creation entry"}
              title={zh ? "一个人的补给，也可以变成一群人的批次" : "One person's supply can become a whole batch"}
              description={
                zh
                  ? `${s.co_create_public_threshold} 人即可开启公开共创页，${s.co_create_label_threshold} 瓶解锁标签与礼盒主题，${s.co_create_flavor_threshold} 瓶解锁风味方向评审。`
                  : `${s.co_create_public_threshold} people open a public co-creation page, ${s.co_create_label_threshold} bottles unlock label and gift-box theming, and ${s.co_create_flavor_threshold} bottles unlock flavor-direction review.`
              }
            />
            <ButtonLink href="/co-create" variant="supply">
              {zh ? "进入共创池" : "Enter the pool"}
            </ButtonLink>
          </div>
          <div className="mt-8 grid gap-4 lg:grid-cols-2">
            {projects.map((p) => (
              <Link key={p.id} href={`/co-create/${p.id}`}>
                <Card hover className="h-full">
                  <div className="flex flex-wrap gap-2">
                    {p.emotion_tags.map((t) => (
                      <Tag key={t} tone="supply">
                        {t}
                      </Tag>
                    ))}
                  </div>
                  <p className="font-display mt-3 text-lg text-porcelain">{p.title}</p>
                  <p className="mt-2 line-clamp-2 text-sm text-mist">{p.concept}</p>
                  <div className="mt-4 space-y-1.5">
                    <div className="flex justify-between text-xs text-mist">
                      <span>
                        {p.current_quantity}/{p.target_quantity} {zh ? "已预订" : "reserved"}
                      </span>
                      <span>
                        {p.supporters} {zh ? "位支持者" : "supporters"}
                      </span>
                    </div>
                    <ProgressBar value={p.current_quantity} max={p.target_quantity} tone="supply" />
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        </Section>
      </div>

      {/* Membership entry */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "核心序列" : "Core Sequence"}
          title={zh ? "生成得更多，保存得更深" : "Generate more, keep more"}
          description={
            zh
              ? "免费账号每天都有生成额度；核心序列为高频创作者准备了更大的月度提案池与更深的档案能力。"
              : "A free account renews its generation allowance daily; Core Sequence adds larger monthly proposal pools and deeper archive powers for frequent creators."
          }
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-3">
          <Card>
            <p className="text-xs uppercase tracking-wider text-mist">{zh ? "免费" : "Free"}</p>
            <p className="font-display mt-1 text-2xl text-porcelain">
              {s.free_daily_chat} <span className="text-sm text-mist">{zh ? "次 / 天" : "/ day"}</span>
            </p>
            <p className="mt-2 text-sm leading-relaxed text-mist">
              {zh ? "每日对话与基础提案额度，注册即得。" : "Daily chats and basic proposals, included with registration."}
            </p>
          </Card>
          <Card className="border-supply/25">
            <p className="text-xs uppercase tracking-wider text-supply">Lite · ¥{s.lite_price_month}/{zh ? "月" : "mo"}</p>
            <p className="font-display mt-1 text-2xl text-porcelain">
              {s.lite_monthly_proposals} <span className="text-sm text-mist">{zh ? "提案 / 月" : "proposals / mo"}</span>
            </p>
            <p className="mt-2 text-sm leading-relaxed text-mist">
              {zh
                ? `每日 ${s.lite_daily_chat} 次对话，月度提案池与档案入口。`
                : `${s.lite_daily_chat} chats a day, a monthly proposal pool, and Reserve access.`}
            </p>
          </Card>
          <Card className="border-supply/25">
            <p className="text-xs uppercase tracking-wider text-supply">Pro · ¥{s.pro_price_month}/{zh ? "月" : "mo"}</p>
            <p className="font-display mt-1 text-2xl text-porcelain">
              {s.pro_monthly_proposals} <span className="text-sm text-mist">{zh ? "提案 / 月" : "proposals / mo"}</span>
            </p>
            <p className="mt-2 text-sm leading-relaxed text-mist">
              {zh
                ? `每日 ${s.pro_daily_chat} 次对话，导出、多版本创作与礼宾通道。`
                : `${s.pro_daily_chat} chats a day, exports, multi-version creation, and the concierge channel.`}
            </p>
          </Card>
        </div>
        <div className="mt-6">
          <ButtonLink href="/membership" variant="supply">
            {zh ? "查看核心序列 →" : "See Core Sequence →"}
          </ButtonLink>
        </div>
      </Section>
    </>
  );
}
