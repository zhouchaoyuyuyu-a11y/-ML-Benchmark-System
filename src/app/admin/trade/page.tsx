import ReviewActions from "@/components/admin/ReviewActions";
import { StatusPill, Tag } from "@/components/ui";
import { db } from "@/lib/store";
import type { ConciergeLead, TradeRequest } from "@/lib/types";

const th = "border-b border-hairline px-3 py-2 text-left text-xs uppercase tracking-wider text-mist";
const td = "px-3 py-2.5";

const requestTone: Record<TradeRequest["request_type"], "default" | "gold" | "supply" | "jade" | "ember"> = {
  quote: "default",
  authorization: "jade",
  enterprise: "gold",
  collaboration: "supply",
  replenishment: "jade",
};

const channelTone: Record<ConciergeLead["channel"], "default" | "gold" | "supply" | "jade" | "ember"> = {
  maison: "gold",
  trade: "default",
  wechat: "jade",
  co_create: "supply",
  membership: "supply",
};

export default async function AdminTradePage() {
  const data = db();
  const userById = new Map(data.users.map((u) => [u.id, u]));
  const requests = [...data.trade_requests].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const leads = [...data.concierge_leads].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const pending = requests.filter((r) => r.human_review_status === "pending").length;

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">Trade & Inquiries</h1>
      <p className="mt-1 text-sm text-mist">
        Quotes, enterprise programs, collaborations, and replenishments — {requests.length} requests,{" "}
        {pending} awaiting human review. Nothing here ships on AI output alone: every request passes
        compliance screening, human review, and quotation before delivery.
      </p>

      <div className="zx-card mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className={th}>ID</th>
              <th className={th}>Requester</th>
              <th className={th}>Type</th>
              <th className={th}>Qty</th>
              <th className={th}>Budget</th>
              <th className={th}>Region</th>
              <th className={th}>Deadline</th>
              <th className={th}>Invoice</th>
              <th className={th}>Compliance</th>
              <th className={th}>Human review</th>
              <th className={th}>Quote</th>
              <th className={th}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {requests.map((r) => (
              <tr key={r.id} className="border-b border-hairline/50 last:border-0">
                <td className={`${td} whitespace-nowrap font-mono text-xs text-mist`}>{r.id}</td>
                <td className={`${td} whitespace-nowrap`}>
                  <p className="text-porcelain">{userById.get(r.user_id)?.nickname ?? r.user_id}</p>
                  {r.organization && <p className="text-[10px] text-mist/70">{r.organization}</p>}
                </td>
                <td className={td}>
                  <Tag tone={requestTone[r.request_type]}>{r.request_type}</Tag>
                </td>
                <td className={`${td} whitespace-nowrap text-porcelain`}>{r.quantity}</td>
                <td className={`${td} whitespace-nowrap text-porcelain`}>{r.budget}</td>
                <td className={`${td} whitespace-nowrap text-mist`}>{r.delivery_region ?? "—"}</td>
                <td className={`${td} whitespace-nowrap text-mist`}>{r.deadline ?? "open"}</td>
                <td className={td}>
                  {r.invoice_required ? (
                    <span className="text-jade">✓</span>
                  ) : (
                    <span className="text-mist/50">—</span>
                  )}
                </td>
                <td className={td}>
                  <StatusPill status={r.compliance_status} />
                </td>
                <td className={td}>
                  <StatusPill status={r.human_review_status} />
                </td>
                <td className={td}>
                  <StatusPill status={r.quote_status} />
                </td>
                <td className={td}>
                  <ReviewActions
                    targetType="trade_request"
                    targetId={r.id}
                    actions={["approve", "reject", "revision", "escalate", "compliance_risk", "infeasible"]}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section className="mt-10">
        <h2 className="font-display text-lg text-porcelain">Human concierge leads</h2>
        <p className="mt-1 text-xs text-mist">
          Scenarios and budgets left with the human concierge across Maison, trade, WeChat, co-creation,
          and membership channels — {leads.length} leads in the pipeline.
        </p>
        <div className="zx-card mt-3 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className={th}>Name</th>
                <th className={th}>Organization</th>
                <th className={th}>Contact</th>
                <th className={th}>Channel</th>
                <th className={th}>Scenario</th>
                <th className={th}>Budget</th>
                <th className={th}>Status</th>
              </tr>
            </thead>
            <tbody>
              {leads.map((l) => (
                <tr key={l.id} className="border-b border-hairline/50 last:border-0">
                  <td className={`${td} whitespace-nowrap text-porcelain`}>{l.name}</td>
                  <td className={`${td} whitespace-nowrap text-mist`}>{l.organization ?? "—"}</td>
                  <td className={`${td} whitespace-nowrap text-mist`}>{l.contact}</td>
                  <td className={td}>
                    <Tag tone={channelTone[l.channel]}>{l.channel.replace(/_/g, " ")}</Tag>
                  </td>
                  <td className={`${td} min-w-64 text-mist`}>{l.scenario}</td>
                  <td className={`${td} whitespace-nowrap text-porcelain`}>{l.budget}</td>
                  <td className={td}>
                    <StatusPill status={l.status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-8 pb-6">
        <div className="zx-card p-4">
          <p className="text-sm text-porcelain">Enterprise pipeline</p>
          <p className="mt-2 text-xs leading-relaxed text-mist">
            Request → compliance screen → human review → quotation → sample path → staged delivery.
            Compliance screening covers age and region rules for alcohol; human review confirms
            supply-chain feasibility on standard base liquids; quotations and deposits are recorded in{" "}
            <span className="text-porcelain">Orders & Payments</span>. Marking a request{" "}
            <span className="text-ember">compliance risk</span> or{" "}
            <span className="text-ember">supply-chain infeasible</span> halts quotation until a concierge
            resolves it with the requester.
          </p>
        </div>
      </section>
    </div>
  );
}
