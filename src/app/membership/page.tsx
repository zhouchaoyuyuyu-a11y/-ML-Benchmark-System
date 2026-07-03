import type { Metadata } from "next";
import MembershipClient from "./MembershipClient";
import { ButtonLink, Card, Notice, PageHero, Section, SectionHeader, Tag } from "@/components/ui";
import { getSessionUser } from "@/lib/auth";
import { orderWorld } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Core Sequence — membership plans and Order Energy",
  description:
    "Join the ZOTAIX Core Sequence: daily Order Energy for AI concierge chats, monthly structured proposals, long-term memory, label exports, co-creation rights, and concierge priority — Lite, Pro, and Enterprise.",
  path: "/membership",
});

export default async function MembershipPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const s = data.settings;
  const user = await getSessionUser();

  const costTiers = [
    {
      icon: "◐",
      en: "Low-cost · daily chat",
      zh_t: "低成本 · 日常对话",
      desc: zh
        ? `每次约 200–400 字的轻回应：情绪关键词、轻建议、一句陪伴。消耗 1 点秩序能量，免费序列每天 ${s.free_daily_chat} 点。`
        : `Light replies of roughly 200–400 words: emotional keywords, a gentle suggestion, one line of company. Costs 1 Order Energy; the free sequence carries ${s.free_daily_chat} per day.`,
    },
    {
      icon: "◈",
      en: "Medium · structured proposal",
      zh_t: "中成本 · 结构化提案",
      desc: zh
        ? `完整概念提案：酒体方向、香氛方向、瓶身方向、命名、瓶身文案与数字印记。计入每月提案额度（Lite ${s.lite_monthly_proposals} / Pro ${s.pro_monthly_proposals}）。`
        : `A full concept proposal: liquid direction, fragrance direction, bottle direction, names, label copy, and a digital mark. Draws from the monthly proposal allowance (Lite ${s.lite_monthly_proposals} / Pro ${s.pro_monthly_proposals}).`,
    },
    {
      icon: "◆",
      en: "High-cost · creative generation",
      zh_t: "高成本 · 创意生成",
      desc: zh
        ? "多版本方案对比、图像生成、3D 瓶身概念、PDF 导出与人工礼宾协助。Pro 与企业序列的核心权益，逐项按次核算。"
        : "Multi-version comparisons, image generation, 3D bottle concepts, PDF export, and human concierge assist. The core of Pro and Enterprise sequences, metered per run.",
    },
  ];

  const quotaRows: { label: string; free: string; lite: string; pro: string }[] = [
    {
      label: zh ? "每日轻量对话（秩序能量）" : "Daily light chats (Order Energy)",
      free: String(s.free_daily_chat),
      lite: String(s.lite_daily_chat),
      pro: String(s.pro_daily_chat),
    },
    {
      label: zh ? "每月结构化提案" : "Structured proposals / month",
      free: zh ? "1 次临时体验" : "1 temporary test",
      lite: String(s.lite_monthly_proposals),
      pro: String(s.pro_monthly_proposals),
    },
    {
      label: zh ? "灵感草稿" : "Inspiration drafts",
      free: zh ? "1 次临时体验" : "1 temporary test",
      lite: zh ? "10 份 / 月" : "10 / month",
      pro: zh ? "扩展额度 + 多版本" : "Expanded + multi-version",
    },
    {
      label: zh ? "情绪卡片" : "Emotional cards",
      free: zh ? "1 张基础卡" : "1 basic card",
      lite: zh ? "包含" : "Included",
      pro: zh ? "高清导出" : "High-res export",
    },
    {
      label: zh ? "长期档案记忆" : "Long-term profile memory",
      free: "—",
      lite: zh ? "基础档案" : "Basic profile",
      pro: zh ? "完整长期记忆" : "Full long-term memory",
    },
    {
      label: zh ? "赠予对象档案" : "Recipient profiles",
      free: "—",
      lite: "1",
      pro: zh ? "多个" : "Multiple",
    },
    {
      label: zh ? "瓶身文案导出" : "Label copy export",
      free: "—",
      lite: zh ? "基础导出" : "Basic export",
      pro: zh ? "高清导出" : "High-res export",
    },
    {
      label: zh ? "共创权限" : "Co-creation",
      free: zh ? "浏览公开案例" : "Browse public cases",
      lite: zh ? "加入项目" : "Join projects",
      pro: zh ? "发起项目 + 创始人身份" : "Start projects + founder identity",
    },
    {
      label: zh ? "数字印记" : "Digital Marks",
      free: "—",
      lite: zh ? "包含" : "Included",
      pro: zh ? "包含" : "Included",
    },
    {
      label: zh ? "实体铸造抵扣" : "Physical casting credit",
      free: "—",
      lite: zh ? "小额抵扣" : "Small credit",
      pro: zh ? "抵扣额度" : "Credit included",
    },
    {
      label: zh ? "礼宾优先级" : "Concierge priority",
      free: "—",
      lite: "—",
      pro: zh ? "优先响应" : "Priority response",
    },
  ];

  const faqs: { q: string; a: string }[] = [
    {
      q: zh ? "秩序能量是什么？什么时候恢复？" : "What is Order Energy, and when does it restore?",
      a: zh
        ? `秩序能量是每日 AI 对话额度的秩序世界称呼。每一次轻量对话消耗 1 点，每天零点自动恢复到序列上限（免费 ${s.free_daily_chat} / Lite ${s.lite_daily_chat} / Pro ${s.pro_daily_chat}）。结构化提案走独立的每月额度，不占用每日能量。`
        : `Order Energy is the Order-World name for your daily AI chat allowance. Each light chat costs 1 point, and the pool restores to your sequence ceiling at midnight (Free ${s.free_daily_chat} / Lite ${s.lite_daily_chat} / Pro ${s.pro_daily_chat}). Structured proposals draw from a separate monthly allowance.`,
    },
    {
      q: zh ? "能量用完了会怎样？" : "What happens when my energy runs out?",
      a: zh
        ? "已生成的一切都还在：你可以继续浏览、编辑草稿、查看档案与共创项目。生成会在次日恢复——或者立即跃迁一级，能量上限即时生效。"
        : "Everything you have made stays available: browse, edit drafts, read your archive, and follow co-creation projects. Generation resumes the next day — or leap a level and the higher ceiling applies immediately.",
    },
    {
      q: zh ? "月付和季付可以切换吗？可以取消吗？" : "Can I switch cycles or cancel?",
      a: zh
        ? "可以。序列按周期计费，权益持续到当前周期结束；随时可以在月付与季付之间切换，或升降级——新的额度即时生效，不影响已保存的档案。"
        : "Yes. Sequences bill per cycle and benefits run to the end of the current period. Switch between monthly and quarterly, or move between Lite and Pro, at any time — new allowances apply immediately and saved archives are never affected.",
    },
    {
      q: zh ? "支付是如何处理的？" : "How are payments handled?",
      a: zh
        ? "支持微信支付、支付宝、Stripe 与 PayPal。当某个支付通道未接入时，订单会被完整记录并转入礼宾确认流程——由人工确认后开通权益，你会在订单面板看到对应说明。"
        : "WeChat Pay, Alipay, Stripe, and PayPal are supported. When a payment channel is not connected, your order is recorded in full and routed through the concierge-confirmation flow — a human confirms it and your benefits activate, with the details shown on your order panel.",
    },
    {
      q: zh ? "会员会直接买到实体酒吗？" : "Does membership buy physical bottles directly?",
      a: zh
        ? "不会。ZOTAIX 的原则是先创造对象，再决定是否实体化。会员权益中的铸造抵扣可用于实体铸造，但所有实体交付都需经人工确认、供应链确认、年龄与地区合规审核及最终报价。"
        : "No. The ZOTAIX principle is to create the object first and decide on the physical later. The casting credit in membership applies toward physical casting, but every physical delivery passes human confirmation, supply-chain confirmation, age and region compliance checks, and a final quotation.",
    },
  ];

  return (
    <>
      <PageHero
        eyebrow={zh ? "核心序列 · Core Sequence" : "Core Sequence · 核心序列"}
        title={zh ? "跃迁到属于你的序列" : "Leap to the sequence that fits you"}
        description={
          zh
            ? "核心序列是 ZOTAIX 的会员体系：更多秩序能量、更深的结构化提案、长期记忆、导出与共创权限。先免费创造，需要更多时再跃迁——永远不会有人把购物车推到你面前。"
            : "The Core Sequence is ZOTAIX membership: more Order Energy, deeper structured proposals, long-term memory, exports, and co-creation rights. Create free first, leap when you need more — no one ever pushes a cart at you."
        }
      >
        <ButtonLink href="#plans" variant="gold">
          {zh ? "查看序列方案" : "See the sequences"}
        </ButtonLink>
        <ButtonLink href="/concierge" variant="outline">
          {zh ? "先免费体验礼宾" : "Try the concierge free"}
        </ButtonLink>
      </PageHero>

      {/* Order World vocabulary */}
      <Section className="py-12 sm:py-16">
        <SectionHeader
          eyebrow={zh ? "秩序世界" : "The Order World"}
          title={zh ? "这里的语言略有不同" : "The language here is a little different"}
          description={
            zh
              ? "ZOTAIX 的会员与游戏化区域使用「秩序世界」词汇。它们只是命名，不改变任何权益的实际含义。"
              : "Membership and gamified areas of ZOTAIX speak the Order-World vocabulary. These are names only — they never change what a benefit actually is."
          }
        />
        <div className="mt-8 overflow-x-auto rounded-lg border border-hairline">
          <table className="w-full min-w-[560px] text-left text-sm">
            <thead>
              <tr className="border-b border-hairline bg-obsidian/60">
                <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-mist">
                  {zh ? "通常的说法" : "Plain meaning"}
                </th>
                <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-gold">
                  {zh ? "秩序世界称呼" : "Order-World name"}
                </th>
                <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-mist">中文</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(orderWorld).map(([key, v]) => (
                <tr key={key} className="border-b border-hairline last:border-0">
                  <td className="px-4 py-2.5 text-mist">{v.plain}</td>
                  <td className="px-4 py-2.5 font-medium text-porcelain">{v.order}</td>
                  <td className="px-4 py-2.5 text-mist">{v.zh}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Plans */}
      <div id="plans" className="border-y border-hairline bg-obsidian/40">
        <Section className="py-12 sm:py-16">
          <SectionHeader
            eyebrow={zh ? "序列方案" : "The sequences"}
            title={zh ? "四个序列，一条定制链" : "Four sequences, one customization chain"}
            description={
              zh
                ? "月付或季付随时切换；企业序列不设公开价格，由 Maison 礼宾按项目报价。"
                : "Switch between monthly and quarterly at any time. The Enterprise sequence carries no public price — the Maison concierge quotes per project."
            }
          />
          <div className="mt-8">
            <MembershipClient
              zh={zh}
              signedIn={!!user}
              currentPlan={user?.membership_level ?? "free"}
              pricing={{
                liteMonth: s.lite_price_month,
                liteQuarter: s.lite_price_quarter,
                proMonth: s.pro_price_month,
                proQuarter: s.pro_price_quarter,
              }}
              freeDaily={s.free_daily_chat}
              liteDaily={s.lite_daily_chat}
              proDaily={s.pro_daily_chat}
              liteProposals={s.lite_monthly_proposals}
              proProposals={s.pro_monthly_proposals}
            />
          </div>
        </Section>
      </div>

      {/* Three-tier AI cost */}
      <Section className="py-12 sm:py-16">
        <SectionHeader
          eyebrow={zh ? "为什么分级" : "Why tiers exist"}
          title={zh ? "三档 AI 成本，对应三种深度" : "Three tiers of AI cost, three depths of output"}
          description={
            zh
              ? "AI 生成的成本随深度上升。序列定价直接对应这三档成本，让轻量陪伴保持免费，让深度创作可持续。"
              : "AI generation costs rise with depth. Sequence pricing maps directly onto these three tiers — keeping light companionship free and deep creation sustainable."
          }
        />
        <div className="mt-8 grid gap-4 lg:grid-cols-3">
          {costTiers.map((t) => (
            <Card key={t.en} className="h-full">
              <span className="text-lg text-gold">{t.icon}</span>
              <p className="font-display mt-2 text-base text-porcelain">{zh ? t.zh_t : t.en}</p>
              <p className="mt-2 text-sm leading-relaxed text-mist">{t.desc}</p>
            </Card>
          ))}
        </div>
      </Section>

      {/* Quota table + paywall example */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-12 sm:py-16">
          <SectionHeader
            eyebrow={zh ? "权益一览" : "Allowances at a glance"}
            title={zh ? "免费 · Lite · Pro 对照表" : "Free · Lite · Pro, side by side"}
          />
          <div className="mt-8 overflow-x-auto rounded-lg border border-hairline">
            <table className="w-full min-w-[640px] text-left text-sm">
              <thead>
                <tr className="border-b border-hairline bg-obsidian/60">
                  <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-mist">
                    {zh ? "权益" : "Benefit"}
                  </th>
                  <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-mist">
                    {zh ? "免费" : "Free"}
                  </th>
                  <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-porcelain">Lite</th>
                  <th className="px-4 py-3 text-xs font-semibold uppercase tracking-wider text-gold">Pro</th>
                </tr>
              </thead>
              <tbody>
                {quotaRows.map((r) => (
                  <tr key={r.label} className="border-b border-hairline last:border-0">
                    <td className="px-4 py-2.5 text-porcelain">{r.label}</td>
                    <td className="px-4 py-2.5 text-mist">{r.free}</td>
                    <td className="px-4 py-2.5 text-mist">{r.lite}</td>
                    <td className="px-4 py-2.5 text-porcelain">{r.pro}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-8 max-w-3xl">
            <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-mist">
              {zh ? "示例 · 到达额度上限时你会看到" : "Example · what you see at the limit"}
            </p>
            <Notice tone="gold" title={zh ? "今日秩序能量已用尽" : "Today's Order Energy is spent"}>
              {zh
                ? "能量会在每天零点恢复。已保存的草稿、档案与共创项目不受影响，随时可以浏览与编辑。想要更高的能量上限与每月提案额度，可以跃迁到 Core Sequence Lite 或 Pro。"
                : "Energy restores at midnight. Saved drafts, archives, and co-creation projects stay fully available to browse and edit. For a higher ceiling and a monthly proposal allowance, leap to Core Sequence Lite or Pro."}
              <span className="mt-2 block">
                <Tag tone="gold">{zh ? "权限跃迁 · Permission Leap" : "Permission Leap · 权限跃迁"}</Tag>
              </span>
            </Notice>
          </div>
        </Section>
      </div>

      {/* FAQ */}
      <Section className="py-12 sm:py-16">
        <SectionHeader
          eyebrow="FAQ"
          title={zh ? "关于核心序列的常见问题" : "Common questions about the Core Sequence"}
        />
        <div className="mt-8 max-w-3xl space-y-3">
          {faqs.map((f) => (
            <details key={f.q} className="zx-card group p-5">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm font-medium text-porcelain">
                {f.q}
                <span className="text-gold transition-transform group-open:rotate-45">＋</span>
              </summary>
              <p className="mt-3 text-sm leading-relaxed text-mist">{f.a}</p>
            </details>
          ))}
        </div>
        <div className="mt-10 flex flex-wrap items-center gap-4">
          <ButtonLink href="/maison" variant="outline">
            {zh ? "企业序列 · 咨询 Maison 礼宾" : "Enterprise sequence · ask the Maison concierge"}
          </ButtonLink>
          <ButtonLink href="/legal/membership" variant="ghost">
            {zh ? "会员服务协议 →" : "Membership Service Agreement →"}
          </ButtonLink>
        </div>
      </Section>
    </>
  );
}
