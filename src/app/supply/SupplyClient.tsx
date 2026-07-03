"use client";

import Link from "next/link";
import { useState } from "react";
import ProposalCard from "@/components/ProposalCard";
import { Button, ButtonLink, Notice } from "@/components/ui";
import type { AiResult } from "@/lib/types";

type SupplyMode = "spirit" | "fragrance";

interface Template {
  key: string;
  icon: string;
  label: { en: string; zh: string };
  emotion: { en: string; zh: string };
  scenario: string;
  sample: { en: string; zh: string };
}

const TEMPLATES: Template[] = [
  {
    key: "birthday",
    icon: "✶",
    label: { en: "Birthday", zh: "生日" },
    emotion: { en: "celebration — another orbit completed", zh: "庆祝——又完成一圈公转" },
    scenario: "Birthday",
    sample: {
      en: "My best friend turns 24 next week and keeps pretending not to care.",
      zh: "我最好的朋友下周满 24 岁，还在假装毫不在意。",
    },
  },
  {
    key: "breakup",
    icon: "❋",
    label: { en: "Breakup", zh: "失恋" },
    emotion: { en: "letting go, quiet recovery", zh: "放手，安静复原" },
    scenario: "Breakup recovery",
    sample: {
      en: "We deleted the playlist. I want something that tastes like moving on.",
      zh: "我们删掉了共享歌单。我想要一种“向前走”的味道。",
    },
  },
  {
    key: "exam",
    icon: "◐",
    label: { en: "Exam", zh: "考试" },
    emotion: { en: "persistence under pressure", zh: "高压之下的坚持" },
    scenario: "Exam season",
    sample: {
      en: "Third all-nighter this week. The library hums like a ship at sea.",
      zh: "这周第三个通宵，图书馆嗡嗡作响像一艘远航的船。",
    },
  },
  {
    key: "workplace",
    icon: "✦",
    label: { en: "Workplace", zh: "职场" },
    emotion: { en: "small victories, held composure", zh: "小小的胜利，稳住的体面" },
    scenario: "Workplace",
    sample: {
      en: "Survived the quarter review. Nobody clapped, so I will.",
      zh: "熬过了季度复盘。没人鼓掌，那我自己来。",
    },
  },
  {
    key: "friendship",
    icon: "❖",
    label: { en: "Friendship", zh: "友情" },
    emotion: { en: "loyalty without ceremony", zh: "不需要仪式感的义气" },
    scenario: "Friendship",
    sample: {
      en: "She moved to another city but still answers on the first ring.",
      zh: "她搬去了另一座城市，电话却仍是一响就接。",
    },
  },
  {
    key: "romance",
    icon: "✷",
    label: { en: "Romance", zh: "恋爱" },
    emotion: { en: "new warmth, careful courage", zh: "新鲜的温度，小心翼翼的勇气" },
    scenario: "Romance",
    sample: {
      en: "Third date on Friday. I want to say something without saying it.",
      zh: "周五第三次约会。我想不动声色地表达一点什么。",
    },
  },
];

const SCENARIOS: { value: string; en: string; zh: string }[] = [
  { value: "Birthday", en: "Birthday", zh: "生日" },
  { value: "Breakup recovery", en: "Breakup recovery", zh: "失恋恢复" },
  { value: "Exam season", en: "Exam season", zh: "考试季" },
  { value: "Workplace", en: "Workplace", zh: "职场情绪" },
  { value: "Friendship", en: "Friendship", zh: "友情" },
  { value: "Romance", en: "Romance", zh: "恋爱" },
  { value: "Everyday", en: "Everyday", zh: "日常" },
];

const BUDGETS: { value: string; en: string; zh: string }[] = [
  { value: "≤100 RMB", en: "≤ 100 RMB", zh: "100 元以内" },
  { value: "100–300 RMB", en: "100–300 RMB", zh: "100–300 元" },
  { value: "300–600 RMB", en: "300–600 RMB", zh: "300–600 元" },
];

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-supply focus:outline-none";

interface GenerateResponse {
  ok?: boolean;
  error?: string;
  upgradeHint?: string;
  conversationId?: string;
  result?: AiResult;
}

export default function SupplyClient({ zh = false }: { zh?: boolean }) {
  const [mode, setMode] = useState<SupplyMode>("spirit");
  const [feeling, setFeeling] = useState("");
  const [emotion, setEmotion] = useState("");
  const [scenario, setScenario] = useState("Everyday");
  const [recipient, setRecipient] = useState("");
  const [budget, setBudget] = useState("≤100 RMB");
  const [zeroProof, setZeroProof] = useState(false);
  const [activeTemplate, setActiveTemplate] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [gate, setGate] = useState<{ message: string; hint: string } | null>(null);
  const [result, setResult] = useState<AiResult | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);

  const examLocked = scenario === "Exam season";
  const effectiveZeroProof = zeroProof || examLocked;

  function applyTemplate(t: Template) {
    setActiveTemplate(t.key);
    setEmotion(zh ? t.emotion.zh : t.emotion.en);
    setScenario(t.scenario);
    const untouched =
      feeling.trim() === "" || TEMPLATES.some((x) => x.sample.en === feeling || x.sample.zh === feeling);
    if (untouched) setFeeling(zh ? t.sample.zh : t.sample.en);
    if (t.key === "exam") setZeroProof(true);
  }

  async function generate() {
    if (!feeling.trim()) {
      setError(zh ? "先写一句今天的感受——一句就够。" : "Write one sentence of feeling first — one is enough.");
      return;
    }
    setLoading(true);
    setError(null);
    setGate(null);
    try {
      const style =
        mode === "spirit"
          ? effectiveZeroProof
            ? "zero-proof, 0.0% ABV, completely alcohol-free, sparkling"
            : "low-ABV, gentle, playful supply line"
          : effectiveZeroProof
            ? "alcohol-free fragrance base, skin-friendly, light"
            : "light everyday fragrance, playful supply line";
      const res = await fetch("/api/ai/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          mode,
          message: feeling,
          emotion: emotion || undefined,
          recipient: recipient || undefined,
          scenario,
          budget,
          style,
          locale: zh ? "zh" : "en",
          conversationId: conversationId ?? undefined,
        }),
      });
      const data = (await res.json().catch(() => ({}))) as GenerateResponse;
      if (res.status === 401 || res.status === 429) {
        setGate({
          message:
            data.error ??
            (zh
              ? "结构化补给会存入你的档案，所以生成器需要知道这是谁的档案。"
              : "Structured supplies are saved to an archive, so the generator needs to know whose archive it is."),
          hint: data.upgradeHint ?? "/register",
        });
        return;
      }
      if (!res.ok || !data.ok || !data.result) {
        setError(data.error ?? (zh ? "生成没有成功，请再试一次。" : "That generation did not go through — try again."));
        return;
      }
      setResult(data.result);
      setConversationId(data.conversationId ?? null);
    } catch {
      setError(zh ? "网络波动了一下，请再试一次。" : "The connection wobbled — please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-8">
      {/* Scenario templates */}
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-supply">
          {zh ? "场景模板 · 一键预填" : "Scenario templates · one tap to prefill"}
        </p>
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {TEMPLATES.map((t) => (
            <button
              key={t.key}
              type="button"
              onClick={() => applyTemplate(t)}
              className={`zx-card p-3 text-left transition-colors hover:border-supply ${
                activeTemplate === t.key ? "border-supply/60" : ""
              }`}
            >
              <span className={`text-lg ${activeTemplate === t.key ? "text-supply" : "text-mist"}`}>{t.icon}</span>
              <p className="font-display mt-1 text-sm text-porcelain">{zh ? t.label.zh : t.label.en}</p>
              <p className="mt-0.5 line-clamp-2 text-xs text-mist">{zh ? t.emotion.zh : t.emotion.en}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Generator form */}
      <div className="zx-card p-5 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="font-display text-lg text-porcelain">
            {zh ? "情绪补给生成器" : "The supply generator"}
          </p>
          <div className="inline-flex overflow-hidden rounded-md border border-hairline">
            {(
              [
                { value: "spirit", en: "Emotional spirit", zh: "情绪之酒" },
                { value: "fragrance", en: "Fragrance supply", zh: "香氛补给" },
              ] as { value: SupplyMode; en: string; zh: string }[]
            ).map((m) => (
              <button
                key={m.value}
                type="button"
                onClick={() => setMode(m.value)}
                className={`px-4 py-2 text-xs font-medium transition-colors ${
                  mode === m.value ? "bg-supply/90 text-ink" : "text-mist hover:text-porcelain"
                }`}
              >
                {zh ? m.zh : m.en}
              </button>
            ))}
          </div>
        </div>

        <div className="mt-5 space-y-4">
          <div>
            <label htmlFor="supply-feeling" className="text-xs uppercase tracking-wider text-mist">
              {zh ? "今天的感受 · 一句话" : "One sentence of feeling"}
            </label>
            <input
              id="supply-feeling"
              value={feeling}
              onChange={(e) => setFeeling(e.target.value)}
              placeholder={
                zh ? "例如：熬过了季度复盘，没人鼓掌，那我自己来。" : "e.g. Survived the quarter review. Nobody clapped, so I will."
              }
              className={`mt-1.5 ${inputCls}`}
              maxLength={200}
            />
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            <div>
              <label htmlFor="supply-scenario" className="text-xs uppercase tracking-wider text-mist">
                {zh ? "场景" : "Scenario"}
              </label>
              <select
                id="supply-scenario"
                value={scenario}
                onChange={(e) => {
                  setScenario(e.target.value);
                  if (e.target.value === "Exam season") setZeroProof(true);
                }}
                className={`mt-1.5 ${inputCls}`}
              >
                {SCENARIOS.map((s) => (
                  <option key={s.value} value={s.value}>
                    {zh ? s.zh : s.en}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="supply-recipient" className="text-xs uppercase tracking-wider text-mist">
                {zh ? "给谁" : "Recipient"}
              </label>
              <input
                id="supply-recipient"
                value={recipient}
                onChange={(e) => setRecipient(e.target.value)}
                placeholder={zh ? "自己 / 好友 / 同事 / 恋人…" : "Self / best friend / colleague / partner…"}
                className={`mt-1.5 ${inputCls}`}
                maxLength={80}
              />
            </div>
            <div>
              <label htmlFor="supply-budget" className="text-xs uppercase tracking-wider text-mist">
                {zh ? "预算" : "Budget"}
              </label>
              <select
                id="supply-budget"
                value={budget}
                onChange={(e) => setBudget(e.target.value)}
                className={`mt-1.5 ${inputCls}`}
              >
                {BUDGETS.map((b) => (
                  <option key={b.value} value={b.value}>
                    {zh ? b.zh : b.en}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="rounded-lg border border-supply/25 bg-supply/5 px-4 py-3">
            <label className="flex cursor-pointer items-start gap-3">
              <input
                type="checkbox"
                checked={effectiveZeroProof}
                disabled={examLocked}
                onChange={(e) => setZeroProof(e.target.checked)}
                className="mt-0.5 h-4 w-4 accent-supply"
              />
              <span>
                <span className="text-sm font-medium text-porcelain">
                  {zh ? "零酒精（0.0%）" : "Zero-proof (0.0%)"}
                </span>
                <span className="mt-0.5 block text-xs leading-relaxed text-mist">
                  {zh
                    ? "零酒精补给完全不含酒精——考试季与所有未成年人相关场景固定生成零酒精方案：同样的仪式感，没有度数。"
                    : "Zero-proof supplies contain no alcohol at all. Exam-season and every minor-safe scenario always generate zero-proof — the same ritual, none of the proof."}
                  {examLocked && (
                    <span className="text-supply">
                      {" "}
                      {zh ? "（考试季已自动锁定零酒精）" : "(locked on for exam season)"}
                    </span>
                  )}
                </span>
              </span>
            </label>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button variant="supply" onClick={generate} disabled={loading}>
              {loading
                ? zh
                  ? "调配中…"
                  : "Blending…"
                : mode === "spirit"
                  ? zh
                    ? "生成我的情绪之酒"
                    : "Generate my emotional spirit"
                  : zh
                    ? "生成我的香氛补给"
                    : "Generate my fragrance supply"}
            </Button>
            <p className="text-xs text-mist">
              {zh
                ? "生成结果是创意提案；补给线酒饮默认低度数（≤12% ABV）。"
                : "Results are creative proposals; supply-line spirits default to low ABV (≤12%)."}
            </p>
          </div>
        </div>
      </div>

      {/* States */}
      {loading && (
        <div className="space-y-3">
          <div className="zx-skeleton h-10 w-2/3 rounded-lg" />
          <div className="zx-skeleton h-36 rounded-lg" />
        </div>
      )}

      {error && !loading && <Notice tone="ember">{error}</Notice>}

      {gate && !loading && (
        <Notice tone="supply" title={zh ? "差一个免费账号" : "One free account away"}>
          <p>{gate.message}</p>
          <div className="mt-3 flex flex-wrap items-center gap-3">
            {gate.hint === "/membership" ? (
              <ButtonLink href="/membership" variant="supply">
                {zh ? "查看核心序列" : "See Core Sequence"}
              </ButtonLink>
            ) : (
              <ButtonLink href="/register" variant="supply">
                {zh ? "免费注册" : "Register free"}
              </ButtonLink>
            )}
            <Link href="/login" className="text-xs text-supply hover:underline">
              {zh ? "已有账号？登录 →" : "Already have one? Sign in →"}
            </Link>
          </div>
        </Notice>
      )}

      {result && !loading && (
        <div className="space-y-4">
          {typeof result.quota_remaining === "number" && (
            <p className="text-xs text-mist">
              {zh
                ? `本周期剩余生成额度：${result.quota_remaining}`
                : `Generations remaining this cycle: ${result.quota_remaining}`}
            </p>
          )}
          <ProposalCard proposal={result.proposal} zh={zh} />
          <div className="zx-card flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between">
            <p className="text-sm leading-relaxed text-mist">
              {zh
                ? "喜欢它？情绪卡片、保存与分享按钮就在上方提案里。也可以带着它走得更远："
                : "Like it? The emotional-card, save, and share buttons live on the proposal above. Or take it further:"}
            </p>
            <div className="flex flex-wrap gap-3">
              <ButtonLink href="/co-create" variant="supply">
                {zh ? "加入共创" : "Join co-creation"}
              </ButtonLink>
              <ButtonLink href="/membership" variant="outline">
                {zh ? "核心序列" : "Core Sequence"}
              </ButtonLink>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
