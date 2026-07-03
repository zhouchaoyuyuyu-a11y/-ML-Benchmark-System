import Link from "next/link";
import { Stat, StatusPill, Tag } from "@/components/ui";
import { integrationStatus } from "@/lib/config";
import { dataMode, db } from "@/lib/store";
import type { AiUsageLog } from "@/lib/types";

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

const modules: { href: string; label: string; desc: string }[] = [
  { href: "/admin/users", label: "Users", desc: "Accounts, tiers, and daily quotas" },
  { href: "/admin/profiles", label: "User Profiles", desc: "Self-expression tags and preferences" },
  { href: "/admin/memberships", label: "Memberships", desc: "Core Sequence plans and limits" },
  { href: "/admin/conversations", label: "Conversations", desc: "Concierge sessions and transcripts" },
  { href: "/admin/ai-usage", label: "AI Usage & Costs", desc: "Tokens, cost estimates, quota burn" },
  { href: "/admin/drafts", label: "Object Drafts", desc: "Saved spirits, fragrances, labels" },
  { href: "/admin/designs", label: "Design Versions", desc: "Bottle, label, and packaging payloads" },
  { href: "/admin/reserve", label: "Reserve Records", desc: "ZOTAIX IDs, certificates, aftercare" },
  { href: "/admin/trade", label: "Trade & Inquiries", desc: "Quotes, enterprise, collaborations" },
  { href: "/admin/co-create", label: "Co-Creation Review", desc: "Project review and thresholds" },
  { href: "/admin/orders", label: "Orders & Payments", desc: "Deposits, memberships, concierge queue" },
  { href: "/admin/moderation", label: "Moderation", desc: "Risk queue and review decisions" },
  { href: "/admin/wechat", label: "WeChat", desc: "Official account menu and auto-replies" },
  { href: "/admin/social", label: "Social Media", desc: "Global official account matrix" },
  { href: "/admin/app", label: "App & Downloads", desc: "Versions, changelog, distribution" },
  { href: "/admin/content-calendar", label: "Content Calendar", desc: "Scheduled posts by platform" },
  { href: "/admin/cms", label: "CMS & Homepage", desc: "Announcement and content blocks" },
  { href: "/admin/legal", label: "Legal Pages", desc: "Terms, privacy, compliance documents" },
  { href: "/admin/settings", label: "Global Settings", desc: "Quotas, pricing, co-create thresholds" },
];

export default async function AdminDashboardPage() {
  const data = db();
  const mode = dataMode();

  const members = data.users.filter((u) => u.user_type === "member").length;
  const gathering = data.co_creation_projects.filter((p) => p.status === "gathering").length;
  const pendingModeration = data.moderation_logs.filter((m) => m.review_status === "pending");
  const pendingTrade = data.trade_requests.filter((t) => t.human_review_status === "pending");
  const pendingProjects = data.co_creation_projects.filter((p) => p.review_status === "pending");
  const pendingReviews = pendingModeration.length + pendingTrade.length + pendingProjects.length;
  const awaitingConcierge = data.orders.filter((o) => o.status === "awaiting_concierge").length;

  const cutoff = Date.now() - 30 * 24 * 60 * 60 * 1000;
  const tokens30d = data.ai_usage_logs
    .filter((l) => new Date(l.created_at).getTime() >= cutoff)
    .reduce((sum, l) => sum + l.tokens_used, 0);

  const stats: { label: string; value: string; hint: string }[] = [
    { label: "Users", value: String(data.users.length), hint: "All account types" },
    { label: "Members", value: String(members), hint: "Core Sequence subscribers" },
    { label: "Object drafts", value: String(data.object_drafts.length), hint: "Saved creations" },
    { label: "Reserve records", value: String(data.reserve_records.length), hint: "Archived identities" },
    { label: "Co-creation gathering", value: String(gathering), hint: "Projects collecting reservations" },
    { label: "Pending reviews", value: String(pendingReviews), hint: "Moderation · trade · co-create" },
    { label: "Awaiting concierge", value: String(awaitingConcierge), hint: "Orders needing human confirmation" },
    { label: "AI tokens · 30d", value: tokens30d.toLocaleString("en-US"), hint: "Across all models" },
  ];

  const attention: { id: string; kind: string; title: string; detail: string; href: string }[] = [
    ...pendingModeration.map((m) => ({
      id: m.id,
      kind: "Moderation",
      title: `${m.content_type.replace(/_/g, " ")} · ${m.risk_type.replace(/_/g, " ")} · ${m.risk_level} risk`,
      detail: m.reviewer_note ?? `Object ${m.object_id} is waiting on a review decision.`,
      href: "/admin/moderation",
    })),
    ...pendingTrade.map((t) => ({
      id: t.id,
      kind: "Trade",
      title: `${t.request_type} request · ${t.quantity} units · ${t.budget}`,
      detail: `${t.delivery_region ?? "Region unspecified"} · quote ${t.quote_status.replace(/_/g, " ")} · deadline ${t.deadline ?? "open"}`,
      href: "/admin/trade",
    })),
    ...pendingProjects.map((p) => ({
      id: p.id,
      kind: "Co-creation",
      title: p.title,
      detail: `${p.current_quantity}/${p.target_quantity} reserved · ${p.supporters} supporters · ${p.votes} votes`,
      href: "/admin/co-create",
    })),
  ];

  const userNameById = new Map(data.users.map((u) => [u.id, u.nickname]));
  const actorFor = (l: AiUsageLog): string =>
    l.user_id ? userNameById.get(l.user_id) ?? l.user_id : `Visitor ${l.visitor_id ?? "anonymous"}`;
  const recentUsage = [...data.ai_usage_logs]
    .sort((a, b) => b.created_at.localeCompare(a.created_at))
    .slice(0, 6);

  const integrations = integrationStatus();

  return (
    <div className="max-w-6xl">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="font-display text-2xl text-porcelain">Dashboard</h1>
        {mode === "memory" ? (
          <Tag tone="supply">seeded in-memory store</Tag>
        ) : (
          <Tag tone="jade">database connected</Tag>
        )}
      </div>
      <p className="mt-1 text-sm text-mist">
        Operational overview of people, objects, review queues, and AI consumption across the platform.
        {mode === "memory"
          ? " Data is served from the seeded in-memory store and resets per server instance; set DATABASE_URL to attach Postgres."
          : ""}
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {stats.map((s) => (
          <Stat key={s.label} label={s.label} value={s.value} hint={s.hint} />
        ))}
      </div>

      <section className="mt-10">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="font-display text-lg text-porcelain">Needs attention</h2>
          <Tag tone="gold">{attention.length} pending</Tag>
        </div>
        <p className="mt-1 text-xs text-mist">
          Items waiting on a human decision — every physical outcome passes through this queue.
        </p>
        {attention.length === 0 ? (
          <p className="zx-card mt-3 p-4 text-sm text-mist">All review queues are clear.</p>
        ) : (
          <div className="mt-3 space-y-3">
            {attention.map((a) => (
              <div key={a.id} className="zx-card flex flex-col gap-3 p-4 sm:flex-row sm:items-center">
                <div className="flex shrink-0 items-center gap-2">
                  <Tag tone="gold">{a.kind}</Tag>
                  <StatusPill status="pending" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm text-porcelain">{a.title}</p>
                  <p className="mt-0.5 truncate text-xs text-mist">{a.detail}</p>
                </div>
                <Link href={a.href} className="shrink-0 text-sm text-gold hover:underline">
                  Review →
                </Link>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="mt-10">
        <h2 className="font-display text-lg text-porcelain">Integration status</h2>
        <p className="mt-1 text-xs text-mist">
          Every integration is environment-driven and degrades to a designed fallback. Badges show which
          credentials are live on this deployment.
        </p>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          {integrations.map((i) => (
            <div key={i.key} className="zx-card p-4">
              <div className="flex items-start justify-between gap-3">
                <p className="text-sm text-porcelain">{i.label}</p>
                {i.configured ? <Tag tone="jade">configured</Tag> : <Tag tone="gold">configuration needed</Tag>}
              </div>
              <p className="mt-2 text-xs leading-relaxed text-mist">
                {i.configured ? "Live credentials detected — requests use the external service." : `Active fallback: ${i.fallback}`}
              </p>
              <p className="mt-2 font-mono text-[10px] tracking-wide text-mist/70">{i.envVars.join(" · ")}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-10">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="font-display text-lg text-porcelain">Recent AI usage</h2>
          <Link href="/admin/ai-usage" className="text-sm text-gold hover:underline">
            Full usage & costs →
          </Link>
        </div>
        <div className="zx-card mt-3 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className={th}>Actor</th>
                <th className={th}>Model</th>
                <th className={th}>Action</th>
                <th className={th}>Tokens</th>
                <th className={th}>Date</th>
              </tr>
            </thead>
            <tbody>
              {recentUsage.map((l) => (
                <tr key={l.id} className="border-b border-hairline/50 last:border-0">
                  <td className={`${td} whitespace-nowrap text-porcelain`}>{actorFor(l)}</td>
                  <td className={`${td} whitespace-nowrap font-mono text-xs text-mist`}>{l.model}</td>
                  <td className={td}>
                    <Tag tone={l.action_type === "chat" ? "supply" : l.action_type === "proposal" ? "gold" : "jade"}>
                      {l.action_type}
                    </Tag>
                  </td>
                  <td className={`${td} text-porcelain`}>{l.tokens_used.toLocaleString("en-US")}</td>
                  <td className={`${td} whitespace-nowrap text-mist`}>{fmtDate(l.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-10 pb-6">
        <h2 className="font-display text-lg text-porcelain">All modules</h2>
        <p className="mt-1 text-xs text-mist">Jump directly into any operational area of the console.</p>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {modules.map((m) => (
            <Link key={m.href} href={m.href} className="block">
              <div className="zx-card zx-card-hover h-full p-4">
                <p className="text-sm text-porcelain">{m.label}</p>
                <p className="mt-1 text-xs leading-relaxed text-mist">{m.desc}</p>
              </div>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
