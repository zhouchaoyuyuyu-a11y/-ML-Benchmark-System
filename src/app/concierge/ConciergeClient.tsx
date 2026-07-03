"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import type { FormEvent } from "react";
import ProposalCard from "@/components/ProposalCard";
import { Button, Notice } from "@/components/ui";
import type { ConceptProposal } from "@/lib/types";

const MODES = [
  {
    key: "daily",
    en: "I just want to talk today",
    zh: "今天只想聊聊",
    phEn: "One sentence about how today feels is enough…",
    phZh: "用一句话说说今天的感觉就够了……",
  },
  {
    key: "gift",
    en: "Help me choose a gift",
    zh: "帮我选一份礼物",
    phEn: "Who is it for, what is the moment, and roughly what budget?",
    phZh: "送给谁？什么时刻？预算大概多少？",
  },
  {
    key: "spirit",
    en: "Help me design a spirit",
    zh: "帮我设计一款酒",
    phEn: "Describe the feeling or the night this spirit should carry…",
    phZh: "描述这瓶酒要承载的情绪，或那个夜晚……",
  },
  {
    key: "fragrance",
    en: "Help me design a fragrance",
    zh: "帮我设计一款香氛",
    phEn: "A place, a season, a memory — what should it smell like?",
    phZh: "一个地方、一个季节、一段记忆——它应该闻起来像什么？",
  },
  {
    key: "copy",
    en: "Help me write bottle copy",
    zh: "帮我写瓶身文案",
    phEn: "What should the bottle say, and to whom?",
    phZh: "这只瓶子想对谁、说什么？",
  },
  {
    key: "style",
    en: "Help me understand my style",
    zh: "帮我理解我的风格",
    phEn: "Colors, music, cities, images you keep returning to…",
    phZh: "说说你反复喜欢的颜色、音乐、城市或画面……",
  },
  {
    key: "recipient",
    en: "Help me create a gift for someone",
    zh: "帮我为某个人创造礼物",
    phEn: "Describe this person — their tastes, your relationship, the occasion…",
    phZh: "描述这个人——TA 的喜好、你们的关系、这个场合……",
  },
  {
    key: "co_create",
    en: "Help me start a co-creation project",
    zh: "帮我发起共创项目",
    phEn: "What idea should a group of people bring into the world together?",
    phZh: "什么样的想法，值得一群人一起把它做出来？",
  },
  {
    key: "enterprise",
    en: "Help me create an enterprise gifting proposal",
    zh: "帮我生成企业礼赠提案",
    phEn: "Audience, quantity, cities, budget per unit, deadline…",
    phZh: "礼赠对象、数量、城市、单件预算、时间节点……",
  },
] as const;

type ModeKey = (typeof MODES)[number]["key"];

function isModeKey(value: string | null): value is ModeKey {
  return MODES.some((m) => m.key === value);
}

interface Turn {
  role: "user" | "assistant";
  content: string;
  proposal?: ConceptProposal;
}

interface Props {
  zh: boolean;
  signedIn: boolean;
}

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

function ConciergeSession({ zh, signedIn }: Props) {
  const params = useSearchParams();
  const urlMode = params.get("mode");
  const [mode, setMode] = useState<ModeKey>(isModeKey(urlMode) ? urlMode : "daily");
  const [message, setMessage] = useState("");
  const [emotion, setEmotion] = useState("");
  const [recipient, setRecipient] = useState("");
  const [scenario, setScenario] = useState("");
  const [budget, setBudget] = useState("");
  const [thread, setThread] = useState<Turn[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [pending, setPending] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [quotaHit, setQuotaHit] = useState<{ message: string; upgradeHint?: string } | null>(null);
  const [quotaRemaining, setQuotaRemaining] = useState<number | null>(null);

  const active = MODES.find((m) => m.key === mode) ?? MODES[0];

  const templateSteps = zh
    ? ["简短情绪回应", "1–3 个关键词", "一条轻建议", "下一步动作"]
    : ["Short emotional response", "1–3 keywords", "Light suggestion", "Next actions"];

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const text = message.trim();
    if (!text || loading) return;
    setError(null);
    setQuotaHit(null);
    setLoading(true);
    setPending(text);
    try {
      const body: Record<string, string> = { mode, message: text, locale: zh ? "zh" : "en" };
      if (conversationId) body.conversationId = conversationId;
      if (mode !== "daily") {
        if (emotion.trim()) body.emotion = emotion.trim();
        if (recipient.trim()) body.recipient = recipient.trim();
        if (scenario.trim()) body.scenario = scenario.trim();
        if (budget.trim()) body.budget = budget.trim();
      }
      const res = await fetch("/api/ai/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = (await res.json()) as {
        ok?: boolean;
        error?: string;
        upgradeHint?: string;
        conversationId?: string;
        result?: { reply: string; proposal: ConceptProposal; quota_remaining: number | null };
      };
      if (res.status === 429) {
        setQuotaHit({
          message:
            data.error ??
            (zh ? "今日对话额度已用完。" : "Today's conversation allowance is used up."),
          upgradeHint: data.upgradeHint,
        });
        return;
      }
      const result = data.result;
      if (!res.ok || !data.ok || !result) {
        setError(
          data.error ??
            (zh ? "礼宾此刻无法回应，请再试一次。" : "The concierge could not respond just now. Please try again.")
        );
        return;
      }
      setConversationId(data.conversationId ?? conversationId);
      setQuotaRemaining(result.quota_remaining);
      setThread((t) => [
        ...t,
        { role: "user", content: text },
        { role: "assistant", content: result.reply, proposal: result.proposal },
      ]);
      setMessage("");
    } catch {
      setError(
        zh
          ? "网络连接异常，请检查网络后重试。"
          : "The concierge could not be reached. Please check your connection and try again."
      );
    } finally {
      setLoading(false);
      setPending("");
    }
  }

  function resetThread() {
    setThread([]);
    setConversationId(null);
    setError(null);
    setQuotaHit(null);
    setQuotaRemaining(null);
  }

  const structuredFields = [
    {
      label: zh ? "情绪" : "Emotion",
      value: emotion,
      set: setEmotion,
      ph: zh ? "如：疲惫但平静" : "e.g. tired but calm",
    },
    {
      label: zh ? "赠予对象" : "Recipient",
      value: recipient,
      set: setRecipient,
      ph: zh ? "如：多年好友" : "e.g. an old friend",
    },
    {
      label: zh ? "场景" : "Scenario",
      value: scenario,
      set: setScenario,
      ph: zh ? "如：周年纪念" : "e.g. anniversary",
    },
    {
      label: zh ? "预算" : "Budget",
      value: budget,
      set: setBudget,
      ph: zh ? "如：600 元" : "e.g. 600 RMB",
    },
  ];

  return (
    <div className="space-y-5">
      {/* Mode chips */}
      <div className="flex flex-wrap gap-2">
        {MODES.map((m) => (
          <button
            key={m.key}
            type="button"
            onClick={() => setMode(m.key)}
            aria-pressed={mode === m.key}
            className={`rounded-full border px-3.5 py-1.5 text-xs transition-colors sm:text-sm ${
              mode === m.key
                ? "border-gold bg-gold/10 text-gold"
                : "border-hairline text-mist hover:border-gold/50 hover:text-porcelain"
            }`}
          >
            {zh ? m.zh : m.en}
          </button>
        ))}
      </div>

      {/* Daily template explainer */}
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5 text-xs text-mist">
        <span className="font-semibold uppercase tracking-[0.2em] text-gold">
          {zh ? "日常模板" : "Daily template"}
        </span>
        {templateSteps.map((s, i) => (
          <span key={s} className="flex items-center gap-2">
            {i > 0 && <span className="text-gold/50">→</span>}
            <span className="rounded-full border border-hairline px-2.5 py-0.5">{s}</span>
          </span>
        ))}
      </div>

      {/* Conversation thread */}
      <div className="zx-card p-4 sm:p-6">
        {thread.length === 0 && !loading ? (
          <div className="py-10 text-center">
            <p className="font-display text-lg text-porcelain">
              {zh ? "礼宾正在听。" : "The concierge is listening."}
            </p>
            <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-mist">
              {zh
                ? `当前模式：「${active.zh}」。写下一句话，礼宾会以情绪回应、关键词与建议作答。`
                : `Current mode: “${active.en}”. Write one sentence and the concierge answers with an emotional response, keywords, and a suggestion.`}
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {thread.map((t, i) =>
              t.role === "user" ? (
                <div key={i} className="flex justify-end">
                  <div className="max-w-[85%] rounded-lg border border-gold/25 bg-gold/5 px-4 py-3 text-sm leading-relaxed text-porcelain sm:max-w-[70%]">
                    {t.content}
                  </div>
                </div>
              ) : (
                <div key={i} className="space-y-3">
                  <div className="max-w-full rounded-lg border border-hairline bg-obsidian px-4 py-3 text-sm leading-relaxed text-porcelain sm:max-w-[85%]">
                    {t.content}
                  </div>
                  {t.proposal && <ProposalCard proposal={t.proposal} zh={zh} />}
                </div>
              )
            )}
            {loading && (
              <>
                <div className="flex justify-end">
                  <div className="max-w-[85%] rounded-lg border border-gold/25 bg-gold/5 px-4 py-3 text-sm leading-relaxed text-porcelain sm:max-w-[70%]">
                    {pending}
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="zx-skeleton h-14 w-full max-w-xl rounded-lg" />
                  <div className="zx-skeleton h-40 w-full rounded-lg" />
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Quota + error states */}
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

      {/* Composer */}
      <form onSubmit={submit} className="space-y-3">
        {mode !== "daily" && (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {structuredFields.map((f) => (
              <label key={f.label} className="block">
                <span className="mb-1 block text-xs uppercase tracking-wider text-mist">{f.label}</span>
                <input
                  value={f.value}
                  onChange={(e) => f.set(e.target.value)}
                  placeholder={f.ph}
                  className={inputCls}
                />
              </label>
            ))}
          </div>
        )}
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          rows={3}
          placeholder={zh ? active.phZh : active.phEn}
          aria-label={zh ? "写给礼宾的信息" : "Message to the concierge"}
          className={inputCls}
        />
        <div className="flex flex-wrap items-center gap-3">
          <Button type="submit" disabled={loading || !message.trim()}>
            {loading ? (zh ? "生成中…" : "Generating…") : zh ? "发送给礼宾" : "Send to the concierge"}
          </Button>
          {thread.length > 0 && (
            <Button type="button" variant="ghost" onClick={resetThread}>
              {zh ? "开始新对话" : "Start a new conversation"}
            </Button>
          )}
          {typeof quotaRemaining === "number" && (
            <span className="text-xs text-mist">
              {zh ? `今日剩余额度：${quotaRemaining}` : `Allowance remaining today: ${quotaRemaining}`}
            </span>
          )}
        </div>
      </form>
    </div>
  );
}

export default function ConciergeClient(props: Props) {
  return (
    <Suspense
      fallback={
        <div className="space-y-4">
          <div className="zx-skeleton h-9 w-full rounded-md" />
          <div className="zx-skeleton h-44 w-full rounded-lg" />
          <div className="zx-skeleton h-24 w-full rounded-md" />
        </div>
      }
    >
      <ConciergeSession {...props} />
    </Suspense>
  );
}
