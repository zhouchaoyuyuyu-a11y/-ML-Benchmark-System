import ReviewActions from "@/components/admin/ReviewActions";
import { Stat, StatusPill, Tag } from "@/components/ui";
import { db } from "@/lib/store";
import type { ModerationLog } from "@/lib/types";

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

const levelTone: Record<ModerationLog["risk_level"], "default" | "gold" | "supply" | "jade" | "ember"> = {
  low: "jade",
  medium: "gold",
  high: "ember",
};

const levelText: Record<ModerationLog["risk_level"], string> = {
  low: "text-jade",
  medium: "text-gold",
  high: "text-ember",
};

const dimensions: { key: string; desc: string }[] = [
  { key: "sensitive", desc: "Politically or socially sensitive content in names, copy, or imagery." },
  { key: "alcohol compliance", desc: "Alcohol marketing rules — age gating, region restrictions, responsible framing." },
  { key: "minor safety", desc: "Content that involves minors or could reach them with alcohol association." },
  { key: "copyright", desc: "Third-party IP appearing in names, label art, or visual styles." },
  { key: "feasibility", desc: "Whether the supply chain can actually produce the direction on standard base liquids." },
  { key: "public display", desc: "Suitability for public archive pages, the inspiration wall, and co-creation listings." },
  { key: "trade eligibility", desc: "Whether a request qualifies for the trade pipeline and quotation." },
  { key: "medical claim", desc: "Health, therapeutic, or functional benefit claims attached to any object." },
  { key: "false promise", desc: "Overpromising — per-bottle new formulas, AI-controlled production, unreviewed shipping." },
  { key: "external transaction", desc: "Attempts to move payment or fulfillment outside the platform's reviewed channels." },
];

export default async function AdminModerationPage() {
  const data = db();
  const logs = [...data.moderation_logs].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const pending = logs.filter((l) => l.review_status === "pending").length;
  const high = logs.filter((l) => l.risk_level === "high").length;
  const approved = logs.filter((l) => l.review_status === "approved").length;

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">Moderation</h1>
      <p className="mt-1 text-sm text-mist">
        The risk queue — {logs.length} logged checks across drafts, projects, and public displays. Every
        flag is resolved by a human decision; approval here is what allows an object to appear publicly or
        move toward a physical outcome.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Logged checks" value={String(logs.length)} hint="All risk dimensions" />
        <Stat label="Pending" value={String(pending)} hint="Awaiting a decision" />
        <Stat label="High risk" value={String(high)} hint="Priority for review" />
        <Stat label="Approved" value={String(approved)} hint="Cleared for display or trade" />
      </div>

      <div className="zx-card mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className={th}>Object</th>
              <th className={th}>Content type</th>
              <th className={th}>Risk type</th>
              <th className={th}>Level</th>
              <th className={th}>Review</th>
              <th className={th}>Reviewer note</th>
              <th className={th}>Created</th>
              <th className={th}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((l) => (
              <tr key={l.id} className="border-b border-hairline/50 last:border-0">
                <td className={`${td} whitespace-nowrap`}>
                  <p className="font-mono text-xs text-porcelain">{l.object_id}</p>
                  <p className="font-mono text-[10px] text-mist/70">{l.id}</p>
                </td>
                <td className={`${td} whitespace-nowrap text-mist`}>
                  {l.content_type.replace(/_/g, " ")}
                </td>
                <td className={td}>
                  <Tag tone={levelTone[l.risk_level]}>{l.risk_type.replace(/_/g, " ")}</Tag>
                </td>
                <td className={`${td} whitespace-nowrap`}>
                  <span className={`text-xs font-medium ${levelText[l.risk_level]}`}>{l.risk_level}</span>
                </td>
                <td className={td}>
                  <StatusPill status={l.review_status} />
                </td>
                <td className={td}>
                  <p className="min-w-56 max-w-72 text-xs leading-relaxed text-mist">
                    {l.reviewer_note ?? "—"}
                  </p>
                </td>
                <td className={`${td} whitespace-nowrap text-mist`}>{fmtDate(l.created_at)}</td>
                <td className={td}>
                  <ReviewActions
                    targetType="moderation_log"
                    targetId={l.id}
                    actions={["approve", "reject", "revision", "escalate"]}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section className="mt-10 pb-6">
        <h2 className="font-display text-lg text-porcelain">Moderation dimensions</h2>
        <p className="mt-1 text-xs text-mist">
          The ten dimensions every generated object and public listing is screened against. Risk-type tags
          in the queue above are colored by level: <span className="text-jade">low</span> ·{" "}
          <span className="text-gold">medium</span> · <span className="text-ember">high</span>.
        </p>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          {dimensions.map((d) => (
            <div key={d.key} className="zx-card p-4">
              <p className="text-sm text-porcelain">{d.key}</p>
              <p className="mt-1.5 text-xs leading-relaxed text-mist">{d.desc}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
