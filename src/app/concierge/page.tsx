import type { Metadata } from "next";
import Link from "next/link";
import { ButtonLink, Card, Notice, PageHero, Section, SectionHeader } from "@/components/ui";
import { getSessionUser } from "@/lib/auth";
import { complianceNotice } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";
import ConciergeClient from "./ConciergeClient";

export const metadata: Metadata = pageMetadata({
  title: "AI Concierge — emotions, gifts, and objects",
  description:
    "Talk to the ZOTAIX AI concierge in nine modes: daily emotional check-ins, gift selection, spirit and fragrance design, bottle copy, style reading, co-creation concepts, and enterprise gifting proposals.",
  path: "/concierge",
});

export default async function ConciergePage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const user = await getSessionUser();
  const settings = db().settings;

  const quotaTiers = [
    {
      title: zh ? "访客" : "Guests",
      value: zh
        ? `每天 ${settings.guest_daily_chat} 次轻量对话`
        : `${settings.guest_daily_chat} lightweight conversations / day`,
      desc: zh
        ? "「今天只想聊聊」无需账号——简短情绪回应、关键词与轻建议。"
        : "The “just talk” mode needs no account — a short emotional response, keywords, and a light suggestion.",
      link: false,
    },
    {
      title: zh ? "注册用户" : "Registered",
      value: zh
        ? `每天 ${settings.free_daily_chat} 次对话额度`
        : `${settings.free_daily_chat} conversations / day`,
      desc: zh
        ? "注册后可保存灵感为草案、保留对话记录，并进入档案与共创。"
        : "Registered accounts save inspirations as drafts, keep conversation history, and enter the archive and co-creation pool.",
      link: false,
    },
    {
      title: zh ? "核心序列会员" : "Core Sequence members",
      value: zh
        ? `Lite ${settings.lite_daily_chat} / Pro ${settings.pro_daily_chat} 次每天`
        : `Lite ${settings.lite_daily_chat} · Pro ${settings.pro_daily_chat} / day`,
      desc: zh
        ? `另含每月 ${settings.lite_monthly_proposals} / ${settings.pro_monthly_proposals} 次结构化提案额度。`
        : `Plus ${settings.lite_monthly_proposals} / ${settings.pro_monthly_proposals} structured proposals per month.`,
      link: true,
    },
  ];

  const signals = [
    {
      en: "Emotion",
      zh: "情绪",
      dEn: "What you are feeling — one word or one paragraph, both work.",
      dZh: "你正在感受什么——一个词或一段话都可以。",
    },
    {
      en: "Recipient",
      zh: "赠予对象",
      dEn: "Yourself, a friend, a partner, a parent, a client.",
      dZh: "你自己、朋友、伴侣、家人或客户。",
    },
    {
      en: "Scenario",
      zh: "场景",
      dEn: "An anniversary, exam week, a farewell, year-end appreciation.",
      dZh: "周年纪念、考试周、告别时刻、年末答谢。",
    },
    {
      en: "Budget",
      zh: "预算",
      dEn: "From a sticker label to a full enterprise gifting program.",
      dZh: "从一张贴纸标签到一整套企业礼赠方案。",
    },
  ];

  const modeGuide = [
    {
      en: "I just want to talk today",
      zh: "今天只想聊聊",
      dEn: "The daily template: a short emotional response, 1–3 keywords, one light suggestion, and next actions.",
      dZh: "日常模板：简短情绪回应、1–3 个关键词、一条轻建议与下一步动作。",
    },
    {
      en: "Help me choose a gift",
      zh: "帮我选一份礼物",
      dEn: "Gift directions matched to person, moment, and budget — with names and label copy.",
      dZh: "按人、时刻与预算匹配礼物方向——附命名与瓶身文案。",
    },
    {
      en: "Help me design a spirit",
      zh: "帮我设计一款酒",
      dEn: "A liquid direction on a standard base, naming candidates, and one line for the bottle.",
      dZh: "基于标准基酒的酒体方向、候选命名与一句瓶身文案。",
    },
    {
      en: "Help me design a fragrance",
      zh: "帮我设计一款香氛",
      dEn: "A fragrance direction — top, heart, base — built from imagery and preference.",
      dZh: "由意象与偏好构成的香氛方向——前调、中调、尾调。",
    },
    {
      en: "Help me write bottle copy",
      zh: "帮我写瓶身文案",
      dEn: "Label copy lines in your voice, ready for a bottle, a card, or a gift box.",
      dZh: "属于你的瓶身文案，可用于瓶身、卡片或礼盒。",
    },
    {
      en: "Help me understand my style",
      zh: "帮我理解我的风格",
      dEn: "A reading of your expression style: colors, imagery, tone, and gifting instincts.",
      dZh: "解读你的表达风格：颜色、意象、语气与送礼直觉。",
    },
    {
      en: "Help me create a gift for someone",
      zh: "帮我为某个人创造礼物",
      dEn: "An object built around one specific person — their tastes, your relationship, the occasion.",
      dZh: "围绕一个具体的人构建对象——TA 的喜好、你们的关系与这个场合。",
    },
    {
      en: "Help me start a co-creation project",
      zh: "帮我发起共创项目",
      dEn: "A concept ready for the co-creation pool: idea, target quantity, and founder benefit.",
      dZh: "可进入共创池的概念：想法、目标数量与发起人权益。",
    },
    {
      en: "Help me create an enterprise gifting proposal",
      zh: "帮我生成企业礼赠提案",
      dEn: "Tiering, unit direction, budget structure, and a sample path for a human concierge to confirm.",
      dZh: "分层、单件方向、预算结构与样品路径，交由人工礼宾确认。",
    },
  ];

  return (
    <>
      <PageHero
        eyebrow={zh ? "AI 礼宾" : "AI Concierge"}
        title={
          zh
            ? "你的 AI 礼宾：情绪、礼物与对象"
            : "Your AI concierge for emotions, gifts, and objects"
        }
        description={
          zh
            ? "一句话就可以开始。礼宾会理解你的情绪、赠予对象、场景与预算，回应关键词、轻建议或一份结构化提案——先创造并保存对象，再决定它成为什么。"
            : "Start with a single sentence. The concierge reads your emotion, recipient, scenario, and budget, then answers with keywords, a light suggestion, or a structured proposal — create and save the object first, then decide what it becomes."
        }
      >
        <ButtonLink href="#session" variant="gold">
          {zh ? "开始对话" : "Start a conversation"}
        </ButtonLink>
        <ButtonLink href="/forge" variant="outline">
          {zh ? "进入 Forge 结构化生成" : "Structured generation in Forge"}
        </ButtonLink>
        <ButtonLink href="/membership" variant="outline">
          {zh ? "查看会员额度" : "See member allowances"}
        </ButtonLink>
      </PageHero>

      {/* Quota explanation strip */}
      <div className="border-b border-hairline bg-obsidian/40">
        <Section className="py-8">
          <div className="grid gap-4 sm:grid-cols-3">
            {quotaTiers.map((t) => (
              <Card key={t.title} className="!p-4">
                <p className="text-xs uppercase tracking-wider text-mist">{t.title}</p>
                <p className="font-display mt-1 text-lg text-porcelain">{t.value}</p>
                <p className="mt-1.5 text-xs leading-relaxed text-mist">{t.desc}</p>
                {t.link && (
                  <Link href="/membership" className="mt-2 inline-block text-xs text-gold hover:underline">
                    {zh ? "对比会员方案 →" : "Compare plans →"}
                  </Link>
                )}
              </Card>
            ))}
          </div>
          <p className="mt-3 text-xs text-mist">
            {zh
              ? "额度每日重置。结构化提案（礼物、酒饮、香氛、企业模式）比日常对话消耗更多额度。"
              : "Allowances reset daily. Structured proposals (gift, spirit, fragrance, and enterprise modes) draw more deeply on your allowance than daily conversations."}
          </p>
        </Section>
      </div>

      {/* Concierge session */}
      <Section id="session" className="py-12 sm:py-16">
        <SectionHeader
          eyebrow={zh ? "对话" : "The session"}
          title={zh ? "选择一个模式，说一句话" : "Pick a mode, say one sentence"}
          description={
            zh
              ? "九个模式对应九种意图。日常模式轻量回应；其余模式生成结构化提案，可直接保存为你的对象草案。"
              : "Nine modes for nine intentions. The daily mode answers lightly; every other mode generates a structured proposal you can save as an object draft."
          }
        />
        <div className="mt-8">
          <ConciergeClient zh={zh} signedIn={!!user} />
        </div>
      </Section>

      {/* What the concierge listens for */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-12 sm:py-16">
          <SectionHeader
            eyebrow={zh ? "四个信号" : "Four signals"}
            title={zh ? "礼宾在听什么" : "What the concierge listens for"}
            description={
              zh
                ? "你不需要填满所有字段——礼宾会从你给出的任何信号开始工作。"
                : "You never need every field — the concierge works from whichever signals you give it."
            }
          />
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {signals.map((s) => (
              <Card key={s.en}>
                <p className="font-display text-base text-porcelain">{zh ? s.zh : s.en}</p>
                <p className="mt-1.5 text-sm leading-relaxed text-mist">{zh ? s.dZh : s.dEn}</p>
              </Card>
            ))}
          </div>
        </Section>
      </div>

      {/* Mode guide */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "九个模式" : "Nine modes"}
          title={zh ? "每个模式返回什么" : "What each mode returns"}
          description={
            zh
              ? "在上方对话区选择模式即可切换。每次回应都附带下一步动作：保存灵感、生成情绪卡片、进入共创或请求人工礼宾。"
              : "Switch modes in the session above. Every response carries next actions: save the inspiration, generate an emotional card, enter co-creation, or request a human concierge."
          }
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {modeGuide.map((m, i) => (
            <Card key={m.en} className="h-full">
              <p className="font-display text-2xl text-gold/60">{String(i + 1).padStart(2, "0")}</p>
              <p className="font-display mt-2 text-base text-porcelain">{zh ? m.zh : m.en}</p>
              <p className="mt-2 text-sm leading-relaxed text-mist">{zh ? m.dZh : m.dEn}</p>
            </Card>
          ))}
        </div>
        <div className="mt-10">
          <Notice tone="gold">{zh ? complianceNotice.zh : complianceNotice.en}</Notice>
        </div>
      </Section>
    </>
  );
}
