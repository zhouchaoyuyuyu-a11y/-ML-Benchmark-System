import Link from "next/link";
import { Tag } from "@/components/ui";
import { db } from "@/lib/store";
import type { MembershipLevel } from "@/lib/types";

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

const planTone: Record<MembershipLevel, "default" | "gold" | "supply" | "jade" | "ember"> = {
  free: "default",
  lite: "supply",
  pro: "gold",
  enterprise: "gold",
};

function mark(v: boolean) {
  return v ? <span className="text-jade">✓</span> : <span className="text-mist/50">—</span>;
}

export default async function AdminMembershipsPage() {
  const data = db();
  const userById = new Map(data.users.map((u) => [u.id, u]));
  const memberships = [...data.memberships].sort((a, b) => a.started_at.localeCompare(b.started_at));
  const s = data.settings;

  const plans = [
    {
      name: "Core Sequence Lite",
      tone: "supply" as const,
      month: s.lite_price_month,
      quarter: s.lite_price_quarter,
      chats: s.lite_daily_chat,
      proposals: s.lite_monthly_proposals,
      perks: "Reserve archive · co-creation join & create",
    },
    {
      name: "Core Sequence Pro",
      tone: "gold" as const,
      month: s.pro_price_month,
      quarter: s.pro_price_quarter,
      chats: s.pro_daily_chat,
      proposals: s.pro_monthly_proposals,
      perks: "Everything in Lite · exports · human concierge channel",
    },
  ];

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">Memberships</h1>
      <p className="mt-1 text-sm text-mist">
        Core Sequence subscriptions — {memberships.length} active plans with their quotas, limits, and
        capability switches. Membership deepens creation; it never gates the ability to be understood.
      </p>

      <div className="zx-card mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className={th}>Member</th>
              <th className={th}>Plan</th>
              <th className={th}>Monthly proposals</th>
              <th className={th}>Daily chats</th>
              <th className={th}>Premium gen</th>
              <th className={th}>Image gen</th>
              <th className={th}>Export</th>
              <th className={th}>Reserve</th>
              <th className={th}>Concierge</th>
              <th className={th}>Started</th>
              <th className={th}>Expires</th>
            </tr>
          </thead>
          <tbody>
            {memberships.map((m) => (
              <tr key={m.id} className="border-b border-hairline/50 last:border-0">
                <td className={`${td} whitespace-nowrap`}>
                  <p className="text-porcelain">{userById.get(m.user_id)?.nickname ?? m.user_id}</p>
                  <p className="font-mono text-[10px] text-mist/70">{m.id}</p>
                </td>
                <td className={td}>
                  <Tag tone={planTone[m.plan]}>{m.plan}</Tag>
                </td>
                <td className={`${td} whitespace-nowrap text-porcelain`}>{m.monthly_quota}</td>
                <td className={`${td} whitespace-nowrap text-porcelain`}>{m.daily_chat_limit}</td>
                <td className={`${td} whitespace-nowrap text-porcelain`}>{m.premium_generation_limit}</td>
                <td className={`${td} whitespace-nowrap text-porcelain`}>{m.image_generation_limit}</td>
                <td className={td}>{mark(m.export_enabled)}</td>
                <td className={td}>{mark(m.reserve_enabled)}</td>
                <td className={td}>{mark(m.concierge_enabled)}</td>
                <td className={`${td} whitespace-nowrap text-mist`}>{fmtDate(m.started_at)}</td>
                <td className={`${td} whitespace-nowrap text-mist`}>{fmtDate(m.expires_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section className="mt-10">
        <h2 className="font-display text-lg text-porcelain">Plan economics</h2>
        <p className="mt-1 text-xs text-mist">
          Live pricing and allowances read from Global Settings — what each Core Sequence plan sells for
          and what it meters.
        </p>
        <div className="mt-3 grid gap-3 lg:grid-cols-2">
          {plans.map((p) => (
            <div key={p.name} className="zx-card p-5">
              <div className="flex items-start justify-between gap-3">
                <p className="font-display text-base text-porcelain">{p.name}</p>
                <Tag tone={p.tone}>{p.tone === "gold" ? "pro" : "lite"}</Tag>
              </div>
              <div className="mt-3 flex flex-wrap items-baseline gap-x-6 gap-y-1">
                <p className="text-porcelain">
                  <span className="font-display text-2xl text-gold">¥{p.month}</span>
                  <span className="ml-1 text-xs text-mist">/ month</span>
                </p>
                <p className="text-porcelain">
                  <span className="font-display text-2xl text-gold">¥{p.quarter}</span>
                  <span className="ml-1 text-xs text-mist">/ quarter</span>
                </p>
              </div>
              <dl className="mt-4 space-y-1.5 text-xs">
                <div>
                  <dt className="inline text-mist">Daily concierge chats — </dt>
                  <dd className="inline text-porcelain">{p.chats}</dd>
                </div>
                <div>
                  <dt className="inline text-mist">Monthly structured proposals — </dt>
                  <dd className="inline text-porcelain">{p.proposals}</dd>
                </div>
                <div>
                  <dt className="inline text-mist">Includes — </dt>
                  <dd className="inline text-porcelain">{p.perks}</dd>
                </div>
              </dl>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-8 pb-6">
        <div className="zx-card p-4">
          <p className="text-sm text-porcelain">Where plan configuration lives</p>
          <p className="mt-2 text-xs leading-relaxed text-mist">
            Prices, chat limits, and proposal allowances are edited in{" "}
            <Link href="/admin/settings" className="text-gold hover:underline">
              Global Settings
            </Link>{" "}
            and apply platform-wide the moment they are saved — the cards above and the membership page both
            read the same values. Individual rows here reflect what each member locked in at subscription
            time.
          </p>
        </div>
      </section>
    </div>
  );
}
