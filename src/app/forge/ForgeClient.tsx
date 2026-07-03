"use client";

import Link from "next/link";
import { useState } from "react";
import type { FormEvent } from "react";
import ProposalCard from "@/components/ProposalCard";
import { Button, Notice } from "@/components/ui";
import type { ConceptProposal, ObjectType } from "@/lib/types";

const OBJECT_TYPES: { key: ObjectType; en: string; zh: string; mode: "spirit" | "fragrance" | "gift" | "enterprise" }[] = [
  { key: "spirit", en: "Spirit", zh: "酒饮", mode: "spirit" },
  { key: "fragrance", en: "Fragrance", zh: "香氛", mode: "fragrance" },
  { key: "bottle", en: "Bottle", zh: "瓶身", mode: "gift" },
  { key: "giftbox", en: "Gift box", zh: "礼盒", mode: "gift" },
  { key: "label", en: "Label", zh: "标签", mode: "gift" },
  { key: "enterprise_gift", en: "Enterprise gift", zh: "企业礼赠", mode: "enterprise" },
];

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

export default function ForgeClient({ zh, signedIn }: { zh: boolean; signedIn: boolean }) {
  const [objectType, setObjectType] = useState<ObjectType>("spirit");
  const [emotion, setEmotion] = useState("");
  const [recipient, setRecipient] = useState("");
  const [scenario, setScenario] = useState("");
  const [budget, setBudget] = useState("");
  const [stylePref, setStylePref] = useState("");
  const [brief, setBrief] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [quotaHit, setQuotaHit] = useState<{ message: string; upgradeHint?: string } | null>(null);
  const [result, setResult] = useState<{ reply: string; proposal: ConceptProposal } | null>(null);
  const [quotaRemaining, setQuotaRemaining] = useState<number | null>(null);

  const selected = OBJECT_TYPES.find((o) => o.key === objectType) ?? OBJECT_TYPES[0];

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (loading) return;
    if (!emotion.trim() && !brief.trim()) {
      setError(
        zh
          ? "至少给 Forge 一种情绪，或一句简述——这是提案的起点。"
          : "Give the Forge at least an emotion or a one-sentence brief — that is where a proposal starts."
      );
      return;
    }
    setError(null);
    setQuotaHit(null);
    setLoading(true);
    try {
      const parts: string[] = [];
      if (emotion.trim()) parts.push(zh ? `情绪：${emotion.trim()}` : `Emotion: ${emotion.trim()}`);
      if (recipient.trim()) parts.push(zh ? `赠予对象：${recipient.trim()}` : `Recipient: ${recipient.trim()}`);
      if (scenario.trim()) parts.push(zh ? `场景：${scenario.trim()}` : `Scenario: ${scenario.trim()}`);
      if (budget.trim()) parts.push(zh ? `预算：${budget.trim()}` : `Budget: ${budget.trim()}`);
      if (stylePref.trim()) parts.push(zh ? `风格偏好：${stylePref.trim()}` : `Style preference: ${stylePref.trim()}`);
      const detail = parts.join(zh ? "；" : "; ");
      const message = brief.trim()
        ? detail
          ? `${brief.trim()}${zh ? `（${detail}）` : ` (${detail})`}`
          : brief.trim()
        : zh
          ? `请为一个「${selected.zh}」对象生成结构化提案。${detail}`
          : `Generate a structured proposal for a ${selected.en.toLowerCase()} object. ${detail}`;

      const body: Record<string, string> = {
        mode: selected.mode,
        message,
        locale: zh ? "zh" : "en",
      };
      if (emotion.trim()) body.emotion = emotion.trim();
      if (recipient.trim()) body.recipient = recipient.trim();
      if (scenario.trim()) body.scenario = scenario.trim();
      if (budget.trim()) body.budget = budget.trim();
      if (stylePref.trim()) body.style = stylePref.trim();

      const res = await fetch("/api/ai/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = (await res.json()) as {
        ok?: boolean;
        error?: string;
        upgradeHint?: string;
        result?: { reply: string; proposal: ConceptProposal; quota_remaining: number | null };
      };
      if (res.status === 429) {
        setQuotaHit({
          message:
            data.error ?? (zh ? "今日生成额度已用完。" : "Today's generation allowance is used up."),
          upgradeHint: data.upgradeHint,
        });
        return;
      }
      const payload = data.result;
      if (!res.ok || !data.ok || !payload) {
        setError(
          data.error ??
            (zh ? "Forge 此刻无法生成，请再试一次。" : "The Forge could not generate just now. Please try again.")
        );
        return;
      }
      setQuotaRemaining(payload.quota_remaining);
      setResult({ reply: payload.reply, proposal: payload.proposal });
    } catch {
      setError(
        zh
          ? "网络连接异常，请检查网络后重试。"
          : "The Forge could not be reached. Please check your connection and try again."
      );
    } finally {
      setLoading(false);
    }
  }

  const fields = [
    {
      label: zh ? "情绪" : "Emotion",
      required: true,
      value: emotion,
      set: setEmotion,
      ph: zh ? "如：季度结束后的安静重建" : "e.g. quiet rebuilding after a hard quarter",
    },
    {
      label: zh ? "赠予对象" : "Recipient",
      required: false,
      value: recipient,
      set: setRecipient,
      ph: zh ? "如：自己 / 伴侣 / 客户" : "e.g. self / partner / clients",
    },
    {
      label: zh ? "场景" : "Scenario",
      required: false,
      value: scenario,
      set: setScenario,
      ph: zh ? "如：深夜工作后的独处" : "e.g. late nights after work",
    },
    {
      label: zh ? "预算" : "Budget",
      required: false,
      value: budget,
      set: setBudget,
      ph: zh ? "如：600 元" : "e.g. 600 RMB",
    },
    {
      label: zh ? "风格偏好" : "Style preference",
      required: false,
      value: stylePref,
      set: setStylePref,
      ph: zh ? "如：克制、东方、哑光" : "e.g. restrained, Eastern, matte",
    },
  ];

  const destinations = [
    {
      title: zh ? "保存灵感 → Design" : "Save Inspiration → Design",
      href: "/design",
      desc: zh
        ? "在提案卡片上点「保存灵感」，对象成为草案，出现在 Design 中，可继续生成命名版本。"
        : "Tap “Save Inspiration” on the proposal card and the object becomes a draft in Design, where named versions grow from it.",
      linkLabel: zh ? "打开 Design →" : "Open Design →",
    },
    {
      title: zh ? "入档 → Reserve" : "Archive → Reserve",
      href: "/reserve",
      desc: zh
        ? "把草案存入档案馆：获得 ZOTAIX ID、QR 绑定与证书页——数字形态本身就是完整的归宿。"
        : "Archive the draft into Reserve: it receives a ZOTAIX ID, QR binding, and certificate page — the digital form is a complete destination in itself.",
      linkLabel: zh ? "打开 Reserve →" : "Open Reserve →",
    },
    {
      title: zh ? "实体铸造 → Trade" : "Physical casting → Trade",
      href: "/trade",
      desc: zh
        ? "如果你希望它成为实物，铸造请求进入 Trade：人工审核、合规检查、样品路径与最终报价。"
        : "If you want it physical, the casting request enters Trade: human review, compliance checks, a sample path, and a final quotation.",
      linkLabel: zh ? "打开 Trade →" : "Open Trade →",
    },
  ];

  const outputs = [
    { en: "Emotional signal", zh: "情绪信号" },
    { en: "1–3 keywords", zh: "1–3 个关键词" },
    { en: "Liquid / fragrance direction", zh: "酒体 / 香氛方向" },
    { en: "Bottle & label direction", zh: "瓶身与标签方向" },
    { en: "Naming candidates", zh: "候选命名" },
    { en: "Label copy", zh: "瓶身文案" },
    { en: "Digital mark", zh: "数字印记" },
    { en: "Next actions", zh: "下一步动作" },
  ];

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      {/* Generator form */}
      <form onSubmit={submit} className="zx-card space-y-4 p-5 sm:p-6">
        <label className="block">
          <span className="mb-1 block text-xs uppercase tracking-wider text-mist">
            {zh ? "对象类型" : "Object type"}
          </span>
          <select
            value={objectType}
            onChange={(e) => setObjectType(e.target.value as ObjectType)}
            className={inputCls}
          >
            {OBJECT_TYPES.map((o) => (
              <option key={o.key} value={o.key}>
                {zh ? o.zh : o.en}
              </option>
            ))}
          </select>
        </label>

        <div className="grid gap-3 sm:grid-cols-2">
          {fields.map((f) => (
            <label key={f.label} className="block">
              <span className="mb-1 block text-xs uppercase tracking-wider text-mist">
                {f.label}
                {f.required && <span className="ml-1 text-gold">*</span>}
              </span>
              <input value={f.value} onChange={(e) => f.set(e.target.value)} placeholder={f.ph} className={inputCls} />
            </label>
          ))}
        </div>

        <label className="block">
          <span className="mb-1 block text-xs uppercase tracking-wider text-mist">
            {zh ? "一句话简述（可选）" : "One-sentence brief (optional)"}
          </span>
          <textarea
            value={brief}
            onChange={(e) => setBrief(e.target.value)}
            rows={2}
            placeholder={
              zh
                ? "如：给刚搬去上海、想念成都的老朋友一瓶带烟火气的酒。"
                : "e.g. A bottle with a trace of hometown warmth, for an old friend who just moved cities."
            }
            className={inputCls}
          />
        </label>

        <div className="flex flex-wrap items-center gap-3">
          <Button type="submit" disabled={loading}>
            {loading ? (zh ? "铸造中…" : "Forging…") : zh ? "生成结构化提案" : "Generate structured proposal"}
          </Button>
          {typeof quotaRemaining === "number" && (
            <span className="text-xs text-mist">
              {zh ? `今日剩余额度：${quotaRemaining}` : `Allowance remaining today: ${quotaRemaining}`}
            </span>
          )}
        </div>
        <p className="text-xs leading-relaxed text-mist">
          {zh
            ? "酒饮与香氛走专属生成模式；瓶身、礼盒与标签走礼物模式；企业礼赠走企业模式并建议接入人工礼宾。"
            : "Spirits and fragrances use their dedicated modes; bottles, gift boxes, and labels use gift mode; enterprise gifts use enterprise mode with a human concierge recommended."}
        </p>
      </form>

      {/* Result + destinations */}
      <div className="space-y-5">
        {quotaHit && (
          <Notice tone="ember" title={zh ? "今日额度已用完" : "Daily allowance reached"}>
            <p>{quotaHit.message}</p>
            {quotaHit.upgradeHint && <p className="mt-1">{quotaHit.upgradeHint}</p>}
            <div className="mt-2 flex flex-wrap gap-4">
              {!signedIn && (
                <Link href="/register" className="text-gold hover:underline">
                  {zh ? "注册获取每日额度 →" : "Register for a daily allowance →"}
                </Link>
              )}
              <Link href="/membership" className="text-gold hover:underline">
                {zh ? "升级核心序列，获得更多额度 →" : "Upgrade to Core Sequence for more →"}
              </Link>
            </div>
          </Notice>
        )}
        {error && <Notice tone="ember">{error}</Notice>}

        {loading ? (
          <div className="space-y-3">
            <div className="zx-skeleton h-14 w-full rounded-lg" />
            <div className="zx-skeleton h-64 w-full rounded-lg" />
          </div>
        ) : result ? (
          <div className="space-y-3">
            <div className="rounded-lg border border-hairline bg-obsidian px-4 py-3 text-sm leading-relaxed text-porcelain">
              {result.reply}
            </div>
            <ProposalCard proposal={result.proposal} zh={zh} />
          </div>
        ) : (
          <div className="zx-card p-5 sm:p-6">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
              {zh ? "Forge 会输出什么" : "What the Forge returns"}
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              {outputs.map((o) => (
                <span key={o.en} className="rounded-full border border-hairline px-2.5 py-0.5 text-xs text-mist">
                  {zh ? o.zh : o.en}
                </span>
              ))}
            </div>
            <p className="mt-4 text-sm leading-relaxed text-mist">
              {zh
                ? "填好左侧输入并点击生成，结构化提案会出现在这里，带着可直接执行的下一步动作。"
                : "Fill the inputs on the left and generate — the structured proposal appears here, carrying next actions you can take immediately."}
            </p>
          </div>
        )}

        {/* What happens to this object */}
        <div className="zx-card p-5 sm:p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
            {zh ? "这个对象会去哪里" : "What happens to this object"}
          </p>
          <div className="mt-4 space-y-4">
            {destinations.map((d) => (
              <div key={d.href} className="border-b border-hairline pb-4 last:border-0 last:pb-0">
                <p className="font-display text-sm text-porcelain">{d.title}</p>
                <p className="mt-1 text-xs leading-relaxed text-mist">{d.desc}</p>
                <Link href={d.href} className="mt-1.5 inline-block text-xs text-gold hover:underline">
                  {d.linkLabel}
                </Link>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
