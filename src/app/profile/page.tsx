import type { Metadata } from "next";
import Link from "next/link";
import ProfileClient from "./ProfileClient";
import { ButtonLink, Card, Notice, PageHero, Section, SectionHeader, Stat, Tag } from "@/components/ui";
import { getSessionUser } from "@/lib/auth";
import { orderWorld, pick, profileNotice } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Order Hub — profile, preferences, and archive",
  description:
    "Your ZOTAIX Order Hub keeps self-expression tags, preferences, recipient profiles, generation history, inspiration drafts, and Reserve records — with privacy controls and one-click export.",
  path: "/profile",
});

const VOCAB_KEYS = ["profile", "quota", "membership", "badge"] as const;

export default async function ProfilePage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const user = await getSessionUser();
  const data = db();

  if (!user) {
    const stores = [
      {
        icon: "◐",
        en: "Self-expression tags",
        zh_t: "自我表达标签",
        desc: zh
          ? "MBTI、星座、血型、年龄段、称呼方式——只作为表达风格的信号，永远不是诊断。"
          : "MBTI, zodiac, blood type, age range, how you like to be addressed — style signals only, never diagnosis.",
      },
      {
        icon: "❖",
        en: "Preferences",
        zh_t: "偏好档案",
        desc: zh
          ? "颜色、香气、酒饮偏好与耐受度、音乐、电影、城市、文学意象、视觉风格、预算区间。"
          : "Colors, scents, alcohol preferences and tolerance, music, films, cities, literary imagery, visual style, budget range.",
      },
      {
        icon: "✶",
        en: "Recipient profiles",
        zh_t: "赠予对象档案",
        desc: zh
          ? "为重要的人建立档案：关系、称呼、偏好、重要日期——礼物提案因此更准。"
          : "Profiles for the people you gift: relation, nickname, preferences, important dates — so gift proposals land.",
      },
      {
        icon: "◈",
        en: "Generation history",
        zh_t: "生成历史",
        desc: zh
          ? "与 AI 礼宾的对话摘要与结构化提案，随时回看、继续或转化为草稿。"
          : "Summaries of your concierge conversations and structured proposals — revisit, continue, or turn them into drafts.",
      },
      {
        icon: "▣",
        en: "Drafts & Reserve records",
        zh_t: "草稿与档案记录",
        desc: zh
          ? "灵感草稿与已入档对象的 ZOTAIX ID、证书与补铸入口都汇聚在这里。"
          : "Inspiration drafts plus archived objects with their ZOTAIX IDs, certificates, and replenishment entries.",
      },
      {
        icon: "◇",
        en: "Privacy & export",
        zh_t: "隐私与导出",
        desc: zh
          ? "记忆开关、私密 / 共创 / 公开三档可见性、一键删除、一键导出完整 JSON 档案。"
          : "Memory toggle, private / co-create / public visibility, one-click delete, one-click JSON export of everything.",
      },
    ];
    return (
      <>
        <PageHero
          eyebrow={zh ? "秩序中枢 · Order Hub" : "Order Hub · 秩序中枢"}
          title={zh ? "一个属于你的秩序中枢" : "A hub for your own order"}
          description={
            zh
              ? "在 ZOTAIX 的秩序世界里，账户是「秩序构筑者」，个人中心是「秩序中枢」。注册后，你的表达标签、偏好、赠予对象与生成历史都会安放于此——AI 礼宾因此更懂你。"
              : "In the ZOTAIX Order World, an account is an Order Builder and the profile center is the Order Hub. Register and your expression tags, preferences, recipients, and generation history live here — so the concierge understands you better each time."
          }
        >
          <ButtonLink href="/register" variant="gold">
            {zh ? "免费注册，开启中枢" : "Register free to open your hub"}
          </ButtonLink>
          <ButtonLink href="/login" variant="outline">
            {zh ? "已有账号？登录" : "Already have an account? Sign in"}
          </ButtonLink>
        </PageHero>

        <Section className="py-12 sm:py-16">
          <SectionHeader
            eyebrow={zh ? "中枢会保存什么" : "What the hub stores"}
            title={zh ? "六类记录，一个档案" : "Six kinds of records, one archive"}
            description={
              zh
                ? "所有内容默认私密，只为你的生成服务。你可以随时修改、导出或删除。"
                : "Everything is private by default and exists only to serve your generations. Edit, export, or delete it any time."
            }
          />
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {stores.map((s) => (
              <Card key={s.en} className="h-full">
                <span className="text-lg text-gold">{s.icon}</span>
                <p className="font-display mt-2 text-base text-porcelain">{zh ? s.zh_t : s.en}</p>
                <p className="mt-1.5 text-sm leading-relaxed text-mist">{s.desc}</p>
              </Card>
            ))}
          </div>
          <div className="mt-8 flex flex-wrap gap-2">
            {VOCAB_KEYS.map((k) => (
              <Tag key={k} tone="gold">
                {orderWorld[k].order} · {orderWorld[k].zh}
              </Tag>
            ))}
          </div>
          <div className="mt-6">
            <Notice tone="gold" title={zh ? "关于表达标签" : "About expression tags"}>
              {pick(locale, profileNotice)}
            </Notice>
          </div>
          <div className="mt-8 flex flex-wrap items-center gap-4">
            <ButtonLink href="/register" variant="gold">
              {zh ? "创建我的秩序中枢" : "Create my Order Hub"}
            </ButtonLink>
            <Link href="/concierge" className="text-sm text-gold hover:underline">
              {zh ? "先以访客身份和礼宾聊聊 →" : "Or talk to the concierge as a guest first →"}
            </Link>
          </div>
        </Section>
      </>
    );
  }

  const today = new Date().toISOString().slice(0, 10);
  const energyUsedToday = data.ai_usage_logs.filter(
    (l) => l.user_id === user.id && l.created_at.slice(0, 10) === today
  ).length;
  const profile = data.user_profiles.find((p) => p.user_id === user.id) ?? null;
  const relationships = data.relationship_profiles.filter((r) => r.user_id === user.id);
  const drafts = data.object_drafts
    .filter((d) => d.user_id === user.id)
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  const reserve = data.reserve_records
    .filter((r) => r.user_id === user.id)
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  const membership = data.memberships.find((m) => m.user_id === user.id) ?? null;
  const conversations = data.conversations
    .filter((c) => c.user_id === user.id)
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
    .slice(0, 8);

  const levelLabel =
    user.membership_level === "pro"
      ? "Core Sequence Pro"
      : user.membership_level === "lite"
        ? "Core Sequence Lite"
        : user.membership_level === "enterprise"
          ? "Enterprise"
          : zh
            ? "自由序列（免费）"
            : "Free sequence";

  return (
    <>
      <PageHero
        eyebrow={zh ? "秩序中枢 · Order Hub" : "Order Hub · 秩序中枢"}
        title={zh ? `欢迎回来，${user.nickname}` : `Welcome back, ${user.nickname}`}
        description={
          zh
            ? "这里是你的秩序中枢：表达标签、偏好、赠予对象、生成历史、草稿与档案记录都在此汇合。所有内容默认私密，可随时导出或删除。"
            : "This is your Order Hub: expression tags, preferences, recipients, generation history, drafts, and Reserve records all converge here. Everything stays private by default and can be exported or deleted at any time."
        }
      >
        <ButtonLink href="/concierge" variant="gold">
          {zh ? "继续与礼宾对话" : "Continue with the concierge"}
        </ButtonLink>
        <ButtonLink href="/membership" variant="outline">
          {zh ? "查看核心序列" : "View Core Sequence"}
        </ButtonLink>
      </PageHero>

      <Section className="py-10 sm:py-14">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Stat
            label={zh ? "今日秩序能量" : "Order Energy today"}
            value={`${energyUsedToday} / ${user.daily_quota}`}
            hint={zh ? "每日零点自动恢复" : "Restores at midnight"}
          />
          <Stat
            label={zh ? "灵感草稿" : "Inspiration drafts"}
            value={String(drafts.length)}
            hint={zh ? "可继续编辑或入档" : "Edit or archive any time"}
          />
          <Stat
            label={zh ? "档案记录" : "Reserve records"}
            value={String(reserve.length)}
            hint={zh ? "每条都有 ZOTAIX ID" : "Each carries a ZOTAIX ID"}
          />
          <Stat
            label={zh ? "核心序列" : "Core Sequence"}
            value={levelLabel}
            hint={
              membership
                ? `${zh ? "有效期至" : "Renews by"} ${membership.expires_at.slice(0, 10)}`
                : zh
                  ? "随时可跃迁一级"
                  : "Leap a level any time"
            }
          />
        </div>

        <div className="mt-10">
          <ProfileClient
            zh={zh}
            profile={profile}
            relationships={relationships}
            drafts={drafts}
            reserve={reserve}
            conversations={conversations}
          />
        </div>
      </Section>
    </>
  );
}
