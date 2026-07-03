import { ProgressBar, Tag } from "@/components/ui";
import { db } from "@/lib/store";
import type { UserType } from "@/lib/types";

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

const typeTone: Record<UserType, "default" | "gold" | "supply" | "jade" | "ember"> = {
  guest: "default",
  registered: "supply",
  member: "jade",
  enterprise: "gold",
  admin: "gold",
};

function mark(v: string) {
  if (v === "yes") return <span className="text-jade">✓</span>;
  if (v === "no") return <span className="text-mist/50">—</span>;
  return <span className="text-xs text-mist">{v}</span>;
}

export default async function AdminUsersPage() {
  const data = db();
  const users = [...data.users].sort((a, b) => a.created_at.localeCompare(b.created_at));
  const s = data.settings;

  const tierColumns = ["Guest", "Registered", "Member", "Enterprise", "Admin"];
  const matrix: { capability: string; values: string[] }[] = [
    {
      capability: "Daily concierge chats",
      values: [`${s.guest_daily_chat}/day trial`, `${s.free_daily_chat}/day`, `${s.lite_daily_chat}–${s.pro_daily_chat}/day`, "Contracted allowance", "Unmetered"],
    },
    {
      capability: "Structured proposals",
      values: ["no", "Within daily allowance", `${s.lite_monthly_proposals}–${s.pro_monthly_proposals}/month`, `${s.pro_monthly_proposals}/month`, "Unmetered"],
    },
    { capability: "Save object drafts", values: ["no", "yes", "yes", "yes", "yes"] },
    { capability: "Reserve archive & certificates", values: ["no", "no", "yes", "yes", "yes"] },
    { capability: "Co-creation · vote", values: ["yes", "yes", "yes", "yes", "yes"] },
    { capability: "Co-creation · join a project", values: ["no", "yes", "yes", "yes", "yes"] },
    { capability: "Co-creation · create a project", values: ["no", "no", "yes", "yes", "yes"] },
    { capability: "Creative tier (multi-version, exports)", values: ["no", "no", "Pro plan", "yes", "yes"] },
    { capability: "Human concierge channel", values: ["no", "no", "Pro plan", "yes", "yes"] },
    { capability: "Enterprise trade & quotations", values: ["no", "no", "no", "yes", "yes"] },
    { capability: "Admin console", values: ["no", "no", "no", "no", "yes"] },
  ];

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">Users</h1>
      <p className="mt-1 text-sm text-mist">
        Every account on the platform — {users.length} total — with tier, Core Sequence level, and today's
        quota consumption.
      </p>

      <div className="zx-card mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className={th}>Nickname</th>
              <th className={th}>Email</th>
              <th className={th}>Type</th>
              <th className={th}>Membership</th>
              <th className={th}>Daily quota</th>
              <th className={th}>Created</th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id} className="border-b border-hairline/50 last:border-0">
                <td className={`${td} whitespace-nowrap text-porcelain`}>{u.nickname}</td>
                <td className={`${td} whitespace-nowrap text-mist`}>{u.email}</td>
                <td className={td}>
                  <Tag tone={typeTone[u.user_type]}>{u.user_type}</Tag>
                </td>
                <td className={`${td} whitespace-nowrap text-porcelain`}>{u.membership_level}</td>
                <td className={td}>
                  <div className="flex items-center gap-2">
                    <span className="whitespace-nowrap text-porcelain">
                      {u.used_quota}
                      <span className="text-mist"> / {u.daily_quota}</span>
                    </span>
                    <div className="w-16 shrink-0">
                      <ProgressBar value={u.used_quota} max={u.daily_quota} />
                    </div>
                  </div>
                </td>
                <td className={`${td} whitespace-nowrap text-mist`}>{fmtDate(u.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section className="mt-10 pb-6">
        <h2 className="font-display text-lg text-porcelain">Tier capability matrix</h2>
        <p className="mt-1 text-xs text-mist">
          What each account type can do. Chat and proposal allowances are read live from Global Settings;
          creation always precedes commerce — no tier gets a cart, and every physical outcome passes human
          review.
        </p>
        <div className="zx-card mt-3 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className={th}>Capability</th>
                {tierColumns.map((t) => (
                  <th key={t} className={th}>
                    {t}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.map((row) => (
                <tr key={row.capability} className="border-b border-hairline/50 last:border-0">
                  <td className={`${td} whitespace-nowrap text-porcelain`}>{row.capability}</td>
                  {row.values.map((v, i) => (
                    <td key={`${row.capability}-${tierColumns[i]}`} className={td}>
                      {mark(v)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          <Tag tone="supply">guest → registered: free account</Tag>
          <Tag tone="jade">registered → member: Core Sequence Lite / Pro</Tag>
          <Tag tone="gold">enterprise: contracted via human concierge</Tag>
        </div>
      </section>
    </div>
  );
}
