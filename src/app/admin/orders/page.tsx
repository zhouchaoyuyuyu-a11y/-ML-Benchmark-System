import { Stat, StatusPill, Tag } from "@/components/ui";
import { integrationStatus } from "@/lib/config";
import { db } from "@/lib/store";
import type { Order, OrderType } from "@/lib/types";

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

const orderTone: Record<OrderType, "default" | "gold" | "supply" | "jade" | "ember"> = {
  membership: "supply",
  label_export: "supply",
  physical_casting: "gold",
  co_creation: "jade",
  premium_deposit: "gold",
  enterprise_project: "gold",
  design_authorization: "jade",
  reserve_replenishment: "jade",
  app_benefit: "supply",
};

const PAYMENT_KEYS = ["stripe", "wechat_pay", "alipay", "paypal"];

function revenueFor(orders: Order[], statuses: Order["status"][]): string {
  const rows = orders.filter((o) => statuses.includes(o.status));
  const cny = rows.filter((o) => o.currency === "CNY").reduce((sum, o) => sum + o.amount, 0);
  const usd = rows.filter((o) => o.currency === "USD").reduce((sum, o) => sum + o.amount, 0);
  const parts: string[] = [];
  if (cny > 0) parts.push(`¥${cny.toLocaleString("en-US")}`);
  if (usd > 0) parts.push(`$${usd.toLocaleString("en-US")}`);
  return parts.length > 0 ? parts.join(" + ") : "¥0";
}

export default async function AdminOrdersPage() {
  const data = db();
  const userById = new Map(data.users.map((u) => [u.id, u]));
  const orders = [...data.orders].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const paymentIntegrations = integrationStatus().filter((i) => PAYMENT_KEYS.includes(i.key));

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">Orders & Payments</h1>
      <p className="mt-1 text-sm text-mist">
        Deposits, memberships, exports, and enterprise projects — {orders.length} orders. Commerce arrives
        at the end of understanding: every order here originates from an object its owner chose to make
        real.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Paid revenue" value={revenueFor(orders, ["paid"])} hint="Settled via live gateways" />
        <Stat
          label="Test-mode volume"
          value={revenueFor(orders, ["test_mode"])}
          hint="Recorded while gateways run in test mode"
        />
        <Stat
          label="Awaiting concierge"
          value={revenueFor(orders, ["awaiting_concierge"])}
          hint="Quoted, pending human confirmation"
        />
        <Stat label="Orders" value={String(orders.length)} hint="All order types" />
      </div>

      <div className="zx-card mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className={th}>ID</th>
              <th className={th}>User</th>
              <th className={th}>Type</th>
              <th className={th}>Title</th>
              <th className={th}>Amount</th>
              <th className={th}>Method</th>
              <th className={th}>Status</th>
              <th className={th}>Reference</th>
              <th className={th}>Created</th>
            </tr>
          </thead>
          <tbody>
            {orders.map((o) => (
              <tr key={o.id} className="border-b border-hairline/50 last:border-0">
                <td className={`${td} whitespace-nowrap font-mono text-xs text-mist`}>{o.id}</td>
                <td className={`${td} whitespace-nowrap text-porcelain`}>
                  {userById.get(o.user_id)?.nickname ?? o.user_id}
                </td>
                <td className={td}>
                  <Tag tone={orderTone[o.order_type]}>{o.order_type.replace(/_/g, " ")}</Tag>
                </td>
                <td className={td}>
                  <p className="max-w-72 truncate text-porcelain" title={o.title}>
                    {o.title}
                  </p>
                </td>
                <td className={`${td} whitespace-nowrap text-porcelain`}>
                  {o.currency === "CNY" ? "¥" : "$"}
                  {o.amount.toLocaleString("en-US")}
                  <span className="ml-1 text-[10px] text-mist">{o.currency}</span>
                </td>
                <td className={`${td} whitespace-nowrap text-mist`}>
                  {o.payment_method.replace(/_/g, " ")}
                </td>
                <td className={td}>
                  <StatusPill status={o.status} />
                </td>
                <td className={`${td} whitespace-nowrap font-mono text-xs text-mist`}>
                  {o.reference ?? "—"}
                </td>
                <td className={`${td} whitespace-nowrap text-mist`}>{fmtDate(o.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section className="mt-10 pb-6">
        <h2 className="font-display text-lg text-porcelain">Payment integrations</h2>
        <p className="mt-1 text-xs text-mist">
          Gateways are environment-driven. When credentials are absent, orders are recorded and routed to
          human concierge confirmation — the designed fallback, so no order is ever lost.
        </p>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          {paymentIntegrations.map((i) => (
            <div key={i.key} className="zx-card p-4">
              <div className="flex items-start justify-between gap-3">
                <p className="text-sm text-porcelain">{i.label}</p>
                {i.configured ? <Tag tone="jade">configured</Tag> : <Tag tone="gold">fallback active</Tag>}
              </div>
              <p className="mt-2 text-xs leading-relaxed text-mist">
                {i.configured
                  ? "Live credentials detected — charges settle through the external gateway."
                  : i.fallback}
              </p>
              <p className="mt-2 font-mono text-[10px] tracking-wide text-mist/70">
                {i.envVars.join(" · ")}
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
