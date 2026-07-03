"use client";

import Link from "next/link";
import { useState } from "react";
import { Button, ButtonLink, Notice, StatusPill, Tag } from "@/components/ui";

type Cycle = "month" | "quarter";
type PayMethod = "wechat_pay" | "alipay" | "stripe" | "paypal";
type PaidPlan = "lite" | "pro";

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

const METHODS: { value: PayMethod; label: string }[] = [
  { value: "wechat_pay", label: "WeChat Pay · 微信支付" },
  { value: "alipay", label: "Alipay · 支付宝" },
  { value: "stripe", label: "Stripe" },
  { value: "paypal", label: "PayPal" },
];

interface SubscribeOrder {
  id: string;
  title: string;
  amount: number;
  currency: string;
  payment_method: string;
  status: string;
  reference?: string;
}

interface SubscribeResponse {
  ok?: boolean;
  error?: string;
  order?: SubscribeOrder;
  note?: string;
}

export interface PlanPricing {
  liteMonth: number;
  liteQuarter: number;
  proMonth: number;
  proQuarter: number;
}

export default function MembershipClient({
  zh = false,
  signedIn,
  currentPlan,
  pricing,
  freeDaily,
  liteDaily,
  proDaily,
  liteProposals,
  proProposals,
}: {
  zh?: boolean;
  signedIn: boolean;
  currentPlan: string;
  pricing: PlanPricing;
  freeDaily: number;
  liteDaily: number;
  proDaily: number;
  liteProposals: number;
  proProposals: number;
}) {
  const [cycle, setCycle] = useState<Cycle>("month");
  const [method, setMethod] = useState<PayMethod>("wechat_pay");
  const [busy, setBusy] = useState<PaidPlan | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [needLogin, setNeedLogin] = useState(false);
  const [result, setResult] = useState<{ order: SubscribeOrder; note?: string } | null>(null);

  async function subscribe(plan: PaidPlan) {
    setBusy(plan);
    setError(null);
    setNeedLogin(false);
    try {
      const res = await fetch("/api/membership/subscribe", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ plan, cycle, method }),
      });
      if (res.status === 401) {
        setNeedLogin(true);
        return;
      }
      const data = (await res.json().catch(() => ({}))) as SubscribeResponse;
      if (!res.ok || !data.ok || !data.order) {
        setError(
          data.error ?? (zh ? "订阅没有成功，请再试一次。" : "That subscription did not go through — please try again.")
        );
        return;
      }
      setResult({ order: data.order, note: data.note });
    } catch {
      setError(zh ? "网络波动了一下，请再试一次。" : "The connection wobbled — please try again.");
    } finally {
      setBusy(null);
    }
  }

  const litePrice = cycle === "month" ? pricing.liteMonth : pricing.liteQuarter;
  const proPrice = cycle === "month" ? pricing.proMonth : pricing.proQuarter;
  const cycleLabel = cycle === "month" ? (zh ? "/月" : "/mo") : zh ? "/季" : "/quarter";
  const methodLabel = (value: string) => METHODS.find((m) => m.value === value)?.label ?? value;

  const freeFeatures = [
    zh ? `每天 ${freeDaily} 次轻量 AI 对话` : `${freeDaily} lightweight AI calls per day`,
    zh ? "1 次临时体验测试" : "1 temporary test",
    zh ? "1 张基础情绪卡片" : "1 basic emotional card",
    zh ? "浏览公开案例与档案" : "Browse public cases and archives",
  ];
  const freeLimits = [
    zh ? "无长期档案记忆" : "No long-term profile memory",
    zh ? "无导出权限" : "No exports",
    zh ? "不可发起共创" : "Cannot start co-creation",
  ];
  const liteFeatures = [
    zh ? `每天 ${liteDaily} 次对话` : `${liteDaily} chats per day`,
    zh ? `每月 ${liteProposals} 次结构化提案` : `${liteProposals} proposals per month`,
    zh ? "每月 10 份灵感草稿" : "10 inspiration drafts per month",
    zh ? "基础档案记忆" : "Basic profile memory",
    zh ? "基础瓶身文案导出" : "Basic label copy export",
    zh ? "加入共创项目" : "Join co-creation projects",
    zh ? "数字印记" : "Digital Marks",
    zh ? "小额实体铸造抵扣" : "Small physical casting credit",
  ];
  const proFeatures = [
    zh ? `每天 ${proDaily} 次对话` : `${proDaily} chats per day`,
    zh ? `每月 ${proProposals} 次结构化提案` : `${proProposals} proposals per month`,
    zh ? "完整长期记忆" : "Full long-term memory",
    zh ? "多个赠予对象档案" : "Multiple recipient profiles",
    zh ? "进阶提案 + 酒体 / 香氛解析" : "Advanced proposals + liquid / fragrance analysis",
    zh ? "高清瓶身文案导出" : "High-res label export",
    zh ? "发起共创 + 创始人身份" : "Start co-creation + founder identity",
    zh ? "礼宾优先响应" : "Concierge priority",
    zh ? "实体铸造抵扣" : "Physical casting credit",
  ];
  const enterpriseFeatures = [
    zh ? "企业礼赠与品牌联名全流程" : "Full enterprise gifting and collaboration flow",
    zh ? "专属人工礼宾与报价" : "Dedicated human concierge and quotation",
    zh ? "样品路径与分批交付" : "Sample paths and staged delivery",
    zh ? "批量档案与补铸管理" : "Batch archives and replenishment",
  ];

  function FeatureList({ items }: { items: string[] }) {
    return (
      <ul className="mt-4 space-y-2">
        {items.map((f) => (
          <li key={f} className="flex items-start gap-2 text-sm leading-relaxed text-mist">
            <span className="mt-0.5 text-gold">✓</span>
            {f}
          </li>
        ))}
      </ul>
    );
  }

  if (result) {
    return (
      <div className="mx-auto max-w-2xl">
        <div className="zx-card p-6 sm:p-8">
          <div className="flex flex-wrap items-center gap-3">
            <h3 className="font-display text-xl text-porcelain">
              {zh ? "序列跃迁已记录" : "Your Permission Leap is recorded"}
            </h3>
            <StatusPill status={result.order.status} />
          </div>
          <dl className="mt-5 space-y-2 text-sm">
            <div className="flex flex-wrap justify-between gap-2 border-b border-hairline pb-2">
              <dt className="text-mist">{zh ? "订单" : "Order"}</dt>
              <dd className="text-porcelain">{result.order.title}</dd>
            </div>
            <div className="flex flex-wrap justify-between gap-2 border-b border-hairline pb-2">
              <dt className="text-mist">{zh ? "金额" : "Amount"}</dt>
              <dd className="font-display text-lg text-gold">
                ¥{result.order.amount} <span className="text-xs text-mist">{result.order.currency}</span>
              </dd>
            </div>
            <div className="flex flex-wrap justify-between gap-2 border-b border-hairline pb-2">
              <dt className="text-mist">{zh ? "支付方式" : "Payment method"}</dt>
              <dd className="text-porcelain">{methodLabel(result.order.payment_method)}</dd>
            </div>
            {result.order.reference && (
              <div className="flex flex-wrap justify-between gap-2">
                <dt className="text-mist">{zh ? "参考编号" : "Reference"}</dt>
                <dd className="font-mono text-xs text-porcelain">{result.order.reference}</dd>
              </div>
            )}
          </dl>
          {result.note && (
            <div className="mt-5">
              <Notice tone="supply" title={zh ? "礼宾确认流程" : "Concierge confirmation"}>
                {result.note}
              </Notice>
            </div>
          )}
          <div className="mt-6 flex flex-wrap gap-3">
            <ButtonLink href="/profile" variant="gold">
              {zh ? "进入我的秩序中枢" : "Open my Order Hub"}
            </ButtonLink>
            <ButtonLink href="/concierge" variant="outline">
              {zh ? "立刻使用新能量" : "Use the new energy now"}
            </ButtonLink>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      {/* Cycle + payment controls */}
      <div className="flex flex-wrap items-end gap-4">
        <div>
          <p className="text-xs uppercase tracking-wider text-mist">{zh ? "计费周期" : "Billing cycle"}</p>
          <div className="mt-2 inline-flex overflow-hidden rounded-md border border-hairline">
            {(
              [
                { value: "month", en: "Monthly", zh_l: "月付" },
                { value: "quarter", en: "Quarterly", zh_l: "季付" },
              ] as { value: Cycle; en: string; zh_l: string }[]
            ).map((c) => (
              <button
                key={c.value}
                type="button"
                onClick={() => setCycle(c.value)}
                className={`px-4 py-2 text-xs font-medium transition-colors ${
                  cycle === c.value ? "bg-gold text-ink" : "text-mist hover:text-porcelain"
                }`}
              >
                {zh ? c.zh_l : c.en}
              </button>
            ))}
          </div>
        </div>
        <label className="block min-w-[220px]">
          <span className="text-xs uppercase tracking-wider text-mist">{zh ? "支付方式" : "Payment method"}</span>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value as PayMethod)}
            className={`mt-2 ${inputCls}`}
          >
            {METHODS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </label>
        {cycle === "quarter" && (
          <Tag tone="jade">
            {zh
              ? `季付更划算：Lite 省 ¥${pricing.liteMonth * 3 - pricing.liteQuarter} · Pro 省 ¥${pricing.proMonth * 3 - pricing.proQuarter}`
              : `Quarterly saves ¥${pricing.liteMonth * 3 - pricing.liteQuarter} on Lite · ¥${pricing.proMonth * 3 - pricing.proQuarter} on Pro`}
          </Tag>
        )}
      </div>

      {error && (
        <div className="mt-5">
          <Notice tone="ember">{error}</Notice>
        </div>
      )}
      {needLogin && (
        <div className="mt-5">
          <Notice tone="gold" title={zh ? "先登录，再跃迁" : "Sign in before you leap"}>
            {zh
              ? "序列权益会绑定到你的秩序中枢，所以需要先登录或注册。"
              : "Sequence benefits bind to your Order Hub, so sign in or register first."}
            <span className="mt-3 flex flex-wrap gap-3">
              <ButtonLink href="/login?next=/membership" variant="gold">
                {zh ? "登录" : "Sign in"}
              </ButtonLink>
              <Link href="/register?next=/membership" className="self-center text-xs text-gold hover:underline">
                {zh ? "还没有账号？免费注册 →" : "No account yet? Register free →"}
              </Link>
            </span>
          </Notice>
        </div>
      )}

      {/* Plan cards */}
      <div className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {/* Free */}
        <div className="zx-card flex h-full flex-col p-5 sm:p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-mist">{zh ? "自由序列" : "Free sequence"}</p>
          <p className="font-display mt-2 text-2xl text-porcelain">¥0</p>
          <p className="mt-1 text-xs text-mist">{zh ? "永久免费" : "Free forever"}</p>
          <FeatureList items={freeFeatures} />
          <ul className="mt-3 space-y-1.5">
            {freeLimits.map((l) => (
              <li key={l} className="flex items-start gap-2 text-xs text-mist/80">
                <span className="mt-0.5">·</span>
                {l}
              </li>
            ))}
          </ul>
          <div className="mt-auto pt-5">
            {signedIn ? (
              currentPlan === "free" ? (
                <Tag tone="jade">{zh ? "当前序列" : "Your current sequence"}</Tag>
              ) : (
                <p className="text-xs text-mist">{zh ? "包含在每个序列中" : "Included in every sequence"}</p>
              )
            ) : (
              <ButtonLink href="/register" variant="outline" className="w-full">
                {zh ? "免费开始" : "Start free"}
              </ButtonLink>
            )}
          </div>
        </div>

        {/* Lite */}
        <div className="zx-card flex h-full flex-col p-5 sm:p-6">
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-porcelain">Core Sequence Lite</p>
            {currentPlan === "lite" && <Tag tone="jade">{zh ? "已激活" : "Active"}</Tag>}
          </div>
          <p className="font-display mt-2 text-2xl text-porcelain">
            ¥{litePrice}
            <span className="text-sm text-mist">{cycleLabel}</span>
          </p>
          <p className="mt-1 text-xs text-mist">
            {cycle === "month"
              ? `¥${pricing.liteQuarter}${zh ? "/季" : "/quarter"}`
              : `¥${pricing.liteMonth}${zh ? "/月" : "/mo"}`}
          </p>
          <FeatureList items={liteFeatures} />
          <div className="mt-auto pt-5">
            <Button variant="outline" className="w-full" onClick={() => void subscribe("lite")} disabled={busy !== null}>
              {busy === "lite" ? (zh ? "开通中…" : "Subscribing…") : zh ? "跃迁到 Lite" : "Leap to Lite"}
            </Button>
          </div>
        </div>

        {/* Pro */}
        <div className="zx-card flex h-full flex-col border-gold/40 p-5 sm:p-6">
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">Core Sequence Pro</p>
            {currentPlan === "pro" ? (
              <Tag tone="jade">{zh ? "已激活" : "Active"}</Tag>
            ) : (
              <Tag tone="gold">{zh ? "最受欢迎" : "Most chosen"}</Tag>
            )}
          </div>
          <p className="font-display mt-2 text-2xl text-porcelain">
            ¥{proPrice}
            <span className="text-sm text-mist">{cycleLabel}</span>
          </p>
          <p className="mt-1 text-xs text-mist">
            {cycle === "month"
              ? `¥${pricing.proQuarter}${zh ? "/季" : "/quarter"}`
              : `¥${pricing.proMonth}${zh ? "/月" : "/mo"}`}
          </p>
          <FeatureList items={proFeatures} />
          <div className="mt-auto pt-5">
            <Button variant="gold" className="w-full" onClick={() => void subscribe("pro")} disabled={busy !== null}>
              {busy === "pro" ? (zh ? "开通中…" : "Subscribing…") : zh ? "跃迁到 Pro" : "Leap to Pro"}
            </Button>
          </div>
        </div>

        {/* Enterprise */}
        <div className="zx-card flex h-full flex-col p-5 sm:p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-porcelain">Enterprise</p>
          <p className="font-display mt-2 text-2xl text-porcelain">{zh ? "按项目报价" : "Quoted per project"}</p>
          <p className="mt-1 text-xs text-mist">{zh ? "不设公开价格" : "No public price"}</p>
          <FeatureList items={enterpriseFeatures} />
          <div className="mt-auto pt-5">
            <ButtonLink href="/maison" variant="outline" className="w-full">
              {zh ? "咨询 Maison 礼宾" : "Ask the Maison concierge"}
            </ButtonLink>
          </div>
        </div>
      </div>

      <p className="mt-6 text-xs leading-relaxed text-mist">
        {zh
          ? "订阅即表示同意《会员服务协议》。序列权益即时生效，按周期持续到期末；实体交付始终另经人工确认与合规审核。"
          : "Subscribing accepts the Membership Service Agreement. Sequence benefits apply immediately and run to the end of the cycle; physical delivery always passes separate human confirmation and compliance review."}
      </p>
    </div>
  );
}
