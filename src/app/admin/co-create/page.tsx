import ReviewActions from "@/components/admin/ReviewActions";
import { ProgressBar, StatusPill, Tag } from "@/components/ui";
import { db } from "@/lib/store";
import type { CoCreationProject } from "@/lib/types";

const th = "border-b border-hairline px-3 py-2 text-left text-xs uppercase tracking-wider text-mist";
const td = "px-3 py-2.5";

const productTone: Record<CoCreationProject["product_type"], "default" | "gold" | "supply" | "jade" | "ember"> = {
  wine: "gold",
  fragrance: "jade",
  bottle: "default",
  giftbox: "supply",
};

export default async function AdminCoCreatePage() {
  const data = db();
  const userById = new Map(data.users.map((u) => [u.id, u]));
  const projects = [...data.co_creation_projects].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  const projectById = new Map(projects.map((p) => [p.id, p]));
  const members = [...data.co_creation_members].sort((a, b) => b.joined_at.localeCompare(a.joined_at));
  const pending = projects.filter((p) => p.review_status === "pending").length;
  const s = data.settings;

  const thresholds: { units: number; unlocks: string }[] = [
    { units: s.co_create_public_threshold, unlocks: "public project page opens" },
    { units: s.co_create_review_threshold, unlocks: "human review round unlocks" },
    { units: s.co_create_label_threshold, unlocks: "label & gift-box theming" },
    { units: s.co_create_flavor_threshold, unlocks: "flavor-direction review" },
    { units: s.co_create_enterprise_threshold, unlocks: "enterprise gifting review" },
    { units: s.co_create_supply_threshold, unlocks: "supply-line production run" },
    { units: s.co_create_partner_threshold, unlocks: "partner program eligibility" },
  ];

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">Co-Creation Review</h1>
      <p className="mt-1 text-sm text-mist">
        Community projects gathering reservations — {projects.length} projects, {pending} awaiting a review
        decision. Depth of customization scales with commitment, and every threshold crossing is reviewed
        by humans before production moves.
      </p>

      <div className="zx-card mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className={th}>Project</th>
              <th className={th}>Creator</th>
              <th className={th}>Product</th>
              <th className={th}>Progress</th>
              <th className={th}>Supporters</th>
              <th className={th}>Votes</th>
              <th className={th}>Status</th>
              <th className={th}>Review</th>
              <th className={th}>Public</th>
              <th className={th}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {projects.map((p) => (
              <tr key={p.id} className="border-b border-hairline/50 last:border-0">
                <td className={td}>
                  <p className="max-w-64 truncate text-porcelain" title={p.title}>
                    {p.title}
                  </p>
                  <p className="font-mono text-[10px] text-mist/70">{p.id}</p>
                </td>
                <td className={`${td} whitespace-nowrap text-mist`}>
                  {userById.get(p.creator_user_id)?.nickname ?? p.creator_user_id}
                </td>
                <td className={td}>
                  <Tag tone={productTone[p.product_type]}>{p.product_type}</Tag>
                </td>
                <td className={td}>
                  <div className="flex items-center gap-2">
                    <span className="whitespace-nowrap text-porcelain">
                      {p.current_quantity}
                      <span className="text-mist"> / {p.target_quantity}</span>
                    </span>
                    <div className="w-32 shrink-0">
                      <ProgressBar value={p.current_quantity} max={p.target_quantity} />
                    </div>
                  </div>
                </td>
                <td className={`${td} whitespace-nowrap text-porcelain`}>{p.supporters}</td>
                <td className={`${td} whitespace-nowrap text-porcelain`}>{p.votes}</td>
                <td className={td}>
                  <StatusPill status={p.status} />
                </td>
                <td className={td}>
                  <StatusPill status={p.review_status} />
                </td>
                <td className={td}>
                  {p.public_visible ? (
                    <span className="text-jade">✓</span>
                  ) : (
                    <span className="text-mist/50">—</span>
                  )}
                </td>
                <td className={td}>
                  <ReviewActions
                    targetType="co_creation_project"
                    targetId={p.id}
                    actions={["approve", "reject", "revision", "feature", "hide"]}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section className="mt-8">
        <h2 className="font-display text-lg text-porcelain">Threshold ladder</h2>
        <p className="mt-1 text-xs text-mist">
          Read live from Global Settings — what each reservation count unlocks for a gathering project.
        </p>
        <div className="zx-card mt-3 flex flex-wrap gap-x-6 gap-y-3 p-4">
          {thresholds.map((t) => (
            <div key={t.units} className="flex items-baseline gap-2">
              <span className="font-display text-lg text-gold">{t.units}</span>
              <span className="text-xs text-mist">{t.unlocks}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="mt-10 pb-6">
        <h2 className="font-display text-lg text-porcelain">Project members</h2>
        <p className="mt-1 text-xs text-mist">
          Founders and participants with reserved quantities — {members.length} memberships across all
          projects. Payments settle only after a project passes review and enters production.
        </p>
        <div className="zx-card mt-3 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className={th}>Project</th>
                <th className={th}>User</th>
                <th className={th}>Role</th>
                <th className={th}>Quantity</th>
                <th className={th}>Payment</th>
              </tr>
            </thead>
            <tbody>
              {members.map((m) => (
                <tr key={m.id} className="border-b border-hairline/50 last:border-0">
                  <td className={td}>
                    <p className="max-w-72 truncate text-porcelain">
                      {projectById.get(m.project_id)?.title ?? m.project_id}
                    </p>
                  </td>
                  <td className={`${td} whitespace-nowrap text-mist`}>
                    {userById.get(m.user_id)?.nickname ?? m.user_id}
                  </td>
                  <td className={td}>
                    <Tag tone={m.role === "founder" ? "gold" : "default"}>{m.role}</Tag>
                  </td>
                  <td className={`${td} whitespace-nowrap text-porcelain`}>{m.quantity}</td>
                  <td className={td}>
                    <StatusPill status={m.payment_status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
