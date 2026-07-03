import Link from "next/link";
import { Notice, ProgressBar, Stat, Tag } from "@/components/ui";
import { db } from "@/lib/store";
import type { AiUsageLog } from "@/lib/types";

function fmtDateTime(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
  });
}

const th = "border-b border-hairline px-3 py-2 text-left text-xs uppercase tracking-wider text-mist";
const td = "px-3 py-2.5";

const actionTone: Record<AiUsageLog["action_type"], "default" | "gold" | "supply" | "jade" | "ember"> = {
  chat: "supply",
  proposal: "gold",
  creative: "jade",
  image: "default",
  export: "default",
};

export default async function AdminAiUsagePage() {
  const data = db();
  const userById = new Map(data.users.map((u) => [u.id, u]));
  const logs = [...data.ai_usage_logs].sort((a, b) => b.created_at.localeCompare(a.created_at));

  const totalTokens = logs.reduce((sum, l) => sum + l.tokens_used, 0);
  const totalCost = logs.reduce((sum, l) => sum + l.cost_estimate, 0);
  const totalQuota = logs.reduce((sum, l) => sum + l.quota_consumed, 0);

  const tiers = ["chat", "proposal", "creative"] as const;
  const byAction = tiers.map((t) => {
    const rows = logs.filter((l) => l.action_type === t);
    return {
      type: t,
      calls: rows.length,
      tokens: rows.reduce((sum, l) => sum + l.tokens_used, 0),
    };
  });

  const modelMap = new Map<string, { calls: number; tokens: number; cost: number }>();
  for (const l of logs) {
    const entry = modelMap.get(l.model) ?? { calls: 0, tokens: 0, cost: 0 };
    entry.calls += 1;
    entry.tokens += l.tokens_used;
    entry.cost += l.cost_estimate;
    modelMap.set(l.model, entry);
  }
  const byModel = [...modelMap.entries()].sort((a, b) => b[1].tokens - a[1].tokens);

  const actorFor = (l: AiUsageLog): string =>
    l.user_id ? userById.get(l.user_id)?.nickname ?? l.user_id : `Visitor ${l.visitor_id ?? "anonymous"}`;

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">AI Usage & Costs</h1>
      <p className="mt-1 text-sm text-mist">
        Every generation is metered here — tokens, cost estimates, and quota consumption per actor, model,
        and action tier.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Total tokens" value={totalTokens.toLocaleString("en-US")} hint="All logged generations" />
        <Stat label="Est. cost" value={`$${totalCost.toFixed(3)}`} hint="USD, provider estimate" />
        <Stat label="Logged calls" value={String(logs.length)} hint="Chat, proposal, creative" />
        <Stat label="Quota consumed" value={String(totalQuota)} hint="Units across all tiers" />
      </div>

      <section className="mt-8">
        <h2 className="font-display text-lg text-porcelain">By action tier</h2>
        <p className="mt-1 text-xs text-mist">
          The three metering tiers the quota engine bills against — each call consumes units on exactly one
          tier.
        </p>
        <div className="mt-3 grid gap-3 sm:grid-cols-3">
          {byAction.map((a) => (
            <Stat
              key={a.type}
              label={a.type}
              value={a.tokens.toLocaleString("en-US")}
              hint={`${a.calls} ${a.calls === 1 ? "call" : "calls"} · tokens`}
            />
          ))}
        </div>
      </section>

      <section className="mt-8">
        <h2 className="font-display text-lg text-porcelain">By model</h2>
        <div className="mt-3 space-y-3">
          {byModel.map(([model, agg]) => (
            <div key={model} className="zx-card p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="font-mono text-sm text-porcelain">{model}</p>
                <p className="text-xs text-mist">
                  {agg.calls} {agg.calls === 1 ? "call" : "calls"} · {agg.tokens.toLocaleString("en-US")}{" "}
                  tokens · ${agg.cost.toFixed(3)}
                </p>
              </div>
              <div className="mt-2">
                <ProgressBar value={agg.tokens} max={totalTokens} />
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-8">
        <h2 className="font-display text-lg text-porcelain">Usage log</h2>
        <div className="zx-card mt-3 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className={th}>Actor</th>
                <th className={th}>Model</th>
                <th className={th}>Action</th>
                <th className={th}>Tokens</th>
                <th className={th}>Est. cost</th>
                <th className={th}>Quota</th>
                <th className={th}>Time (UTC)</th>
              </tr>
            </thead>
            <tbody>
              {logs.map((l) => (
                <tr key={l.id} className="border-b border-hairline/50 last:border-0">
                  <td className={`${td} whitespace-nowrap text-porcelain`}>{actorFor(l)}</td>
                  <td className={`${td} whitespace-nowrap font-mono text-xs text-mist`}>{l.model}</td>
                  <td className={td}>
                    <Tag tone={actionTone[l.action_type]}>{l.action_type}</Tag>
                  </td>
                  <td className={`${td} text-porcelain`}>{l.tokens_used.toLocaleString("en-US")}</td>
                  <td className={`${td} whitespace-nowrap text-mist`}>${l.cost_estimate.toFixed(4)}</td>
                  <td className={`${td} text-mist`}>{l.quota_consumed}</td>
                  <td className={`${td} whitespace-nowrap text-mist`}>{fmtDateTime(l.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-8 pb-6">
        <Notice tone="gold" title="Tier costs and quota configuration">
          Generations bill against one of three tiers: <span className="text-porcelain">chat</span> (light
          conversational replies — guest trial and daily allowances),{" "}
          <span className="text-porcelain">proposal</span> (structured concept proposals — counted against
          Core Sequence monthly proposal quotas), and <span className="text-porcelain">creative</span>{" "}
          (multi-version and export-grade generation — Core Sequence Pro and enterprise). Daily chat limits,
          monthly proposal quotas, and plan pricing are all configured in{" "}
          <Link href="/admin/settings" className="text-gold hover:underline">
            Global Settings
          </Link>
          .
        </Notice>
      </section>
    </div>
  );
}
