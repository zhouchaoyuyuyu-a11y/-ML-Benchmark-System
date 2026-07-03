import { Tag } from "@/components/ui";
import { db } from "@/lib/store";
import type { ConversationType } from "@/lib/types";

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

const th = "border-b border-hairline px-3 py-2 text-left text-xs uppercase tracking-wider text-mist";
const td = "px-3 py-2.5";

const typeTone: Record<ConversationType, "default" | "gold" | "supply" | "jade" | "ember"> = {
  daily: "supply",
  gift: "gold",
  product: "default",
  fragrance: "jade",
  wine: "default",
  enterprise: "gold",
};

export default async function AdminConversationsPage() {
  const data = db();
  const userById = new Map(data.users.map((u) => [u.id, u]));
  const conversations = [...data.conversations].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  const totalTokens = conversations.reduce((sum, c) => sum + c.token_usage, 0);

  const sample = data.conversations[0];
  const sampleMessages = sample ? data.messages.filter((m) => m.conversation_id === sample.id) : [];
  const sampleOwner = sample?.user_id ? userById.get(sample.user_id) : undefined;

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">Conversations</h1>
      <p className="mt-1 text-sm text-mist">
        Concierge sessions across all modes — {conversations.length} conversations,{" "}
        {totalTokens.toLocaleString("en-US")} tokens total. Summaries are model-written; full transcripts
        stay bound to each session.
      </p>

      <div className="zx-card mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className={th}>Actor</th>
              <th className={th}>Type</th>
              <th className={th}>Mode</th>
              <th className={th}>Summary</th>
              <th className={th}>Tokens</th>
              <th className={th}>Updated</th>
            </tr>
          </thead>
          <tbody>
            {conversations.map((c) => {
              const actor = c.user_id
                ? userById.get(c.user_id)?.nickname ?? c.user_id
                : `Visitor ${c.visitor_id ?? "anonymous"}`;
              return (
                <tr key={c.id} className="border-b border-hairline/50 last:border-0">
                  <td className={`${td} whitespace-nowrap`}>
                    <p className="text-porcelain">{actor}</p>
                    <p className="font-mono text-[10px] text-mist/70">{c.id}</p>
                  </td>
                  <td className={td}>
                    <Tag tone={typeTone[c.conversation_type]}>{c.conversation_type}</Tag>
                  </td>
                  <td className={`${td} whitespace-nowrap text-mist`}>{c.mode}</td>
                  <td className={`${td} min-w-72 text-mist`}>{c.summary}</td>
                  <td className={`${td} whitespace-nowrap text-porcelain`}>
                    {c.token_usage.toLocaleString("en-US")}
                  </td>
                  <td className={`${td} whitespace-nowrap text-mist`}>{fmtDate(c.updated_at)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {sample && (
        <section className="mt-10 pb-6">
          <h2 className="font-display text-lg text-porcelain">Transcript sample</h2>
          <p className="mt-1 text-xs text-mist">
            {sampleOwner ? sampleOwner.nickname : `Visitor ${sample.visitor_id ?? "anonymous"}`} ·{" "}
            {sample.conversation_type} mode · {sample.summary}
          </p>
          <div className="mt-4 space-y-4">
            {sampleMessages.map((m) => (
              <div key={m.id} className={m.role === "user" ? "flex justify-end" : "flex justify-start"}>
                <div
                  className={`w-full max-w-xl rounded-lg border px-4 py-3 ${
                    m.role === "user" ? "border-hairline bg-veil" : "border-gold/25 bg-obsidian"
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <span
                      className={`text-[10px] font-semibold uppercase tracking-[0.2em] ${
                        m.role === "assistant" ? "text-gold" : "text-mist"
                      }`}
                    >
                      {m.role}
                    </span>
                    <span className="text-[10px] text-mist/70">{m.token_usage} tokens</span>
                  </div>
                  <p className="mt-1.5 text-sm leading-relaxed text-porcelain">{m.content}</p>

                  {m.structured && (
                    <div className="mt-3 rounded-md border border-gold/20 bg-ink/60 p-3">
                      <div className="flex flex-wrap items-center gap-2">
                        <Tag tone="gold">structured proposal</Tag>
                        <span className="text-xs text-mist">kind · {m.structured.kind}</span>
                      </div>
                      <p className="mt-2 text-sm text-porcelain">{m.structured.emotional_signal}</p>
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {m.structured.keywords.map((k) => (
                          <Tag key={k}>{k}</Tag>
                        ))}
                      </div>
                      <dl className="mt-3 space-y-1.5 text-xs">
                        {m.structured.liquid_direction && (
                          <div>
                            <dt className="inline text-mist">Liquid — </dt>
                            <dd className="inline text-porcelain">{m.structured.liquid_direction}</dd>
                          </div>
                        )}
                        {m.structured.scent_direction && (
                          <div>
                            <dt className="inline text-mist">Scent — </dt>
                            <dd className="inline text-porcelain">{m.structured.scent_direction}</dd>
                          </div>
                        )}
                        {m.structured.bottle_direction && (
                          <div>
                            <dt className="inline text-mist">Bottle — </dt>
                            <dd className="inline text-porcelain">{m.structured.bottle_direction}</dd>
                          </div>
                        )}
                        {m.structured.names && m.structured.names.length > 0 && (
                          <div>
                            <dt className="inline text-mist">Names — </dt>
                            <dd className="inline text-porcelain">{m.structured.names.join(" · ")}</dd>
                          </div>
                        )}
                        {m.structured.digital_mark && (
                          <div>
                            <dt className="inline text-mist">Digital mark — </dt>
                            <dd className="inline text-porcelain">{m.structured.digital_mark}</dd>
                          </div>
                        )}
                      </dl>
                      {m.structured.label_copy && (
                        <p className="font-display mt-3 border-l-2 border-gold/40 pl-3 text-sm italic text-porcelain">
                          “{m.structured.label_copy}”
                        </p>
                      )}
                      {m.structured.next_actions.length > 0 && (
                        <div className="mt-3 flex flex-wrap gap-1.5">
                          {m.structured.next_actions.map((a) => (
                            <Tag key={a} tone="jade">
                              {a.replace(/_/g, " ")}
                            </Tag>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
