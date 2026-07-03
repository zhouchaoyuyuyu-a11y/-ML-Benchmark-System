import ReviewActions from "@/components/admin/ReviewActions";
import { Stat, StatusPill, Tag } from "@/components/ui";
import { db } from "@/lib/store";
import type { ObjectDraft, ObjectType } from "@/lib/types";

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

const typeTone: Record<ObjectType, "default" | "gold" | "supply" | "jade" | "ember"> = {
  spirit: "gold",
  fragrance: "jade",
  bottle: "default",
  giftbox: "supply",
  label: "supply",
  enterprise_gift: "gold",
};

function directionsPreview(d: ObjectDraft): string {
  return [d.liquid_direction, d.scent_direction, d.visual_style, d.label_copy]
    .filter(Boolean)
    .join(" · ");
}

export default async function AdminDraftsPage() {
  const data = db();
  const userById = new Map(data.users.map((u) => [u.id, u]));
  const drafts = [...data.object_drafts].sort((a, b) => b.updated_at.localeCompare(a.updated_at));

  const publicCount = drafts.filter((d) => d.public_visible).length;
  const savedCount = drafts.filter((d) => d.status === "saved").length;
  const reviewedCount = drafts.filter((d) => d.status === "reviewed").length;

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">Object Drafts</h1>
      <p className="mt-1 text-sm text-mist">
        Every saved creation on the platform — {drafts.length} drafts across spirits, fragrances, labels,
        and enterprise gifts. Drafts are the unit of value: they stay digital until their owner chooses a
        physical path.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Total drafts" value={String(drafts.length)} hint="All object types" />
        <Stat label="Saved" value={String(savedCount)} hint="Kept in owner archives" />
        <Stat label="Reviewed" value={String(reviewedCount)} hint="Passed human review" />
        <Stat label="Publicly visible" value={String(publicCount)} hint="Eligible for the inspiration wall" />
      </div>

      <div className="zx-card mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className={th}>Title</th>
              <th className={th}>Owner</th>
              <th className={th}>Type</th>
              <th className={th}>Emotions</th>
              <th className={th}>Directions</th>
              <th className={th}>Status</th>
              <th className={th}>Public</th>
              <th className={th}>Updated</th>
              <th className={th}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {drafts.map((d) => (
              <tr key={d.id} className="border-b border-hairline/50 last:border-0">
                <td className={`${td} whitespace-nowrap`}>
                  <p className="text-porcelain">{d.title}</p>
                  <p className="font-mono text-[10px] text-mist/70">{d.id}</p>
                </td>
                <td className={`${td} whitespace-nowrap text-mist`}>
                  {userById.get(d.user_id)?.nickname ?? d.user_id}
                </td>
                <td className={td}>
                  <Tag tone={typeTone[d.object_type]}>{d.object_type.replace(/_/g, " ")}</Tag>
                </td>
                <td className={td}>
                  <div className="flex max-w-48 flex-wrap gap-1">
                    {d.emotion_tags.map((t) => (
                      <Tag key={t}>{t}</Tag>
                    ))}
                  </div>
                </td>
                <td className={td}>
                  <p className="max-w-64 truncate text-xs text-mist" title={directionsPreview(d)}>
                    {directionsPreview(d)}
                  </p>
                </td>
                <td className={td}>
                  <StatusPill status={d.status} />
                </td>
                <td className={td}>
                  {d.public_visible ? (
                    <span className="text-jade">✓</span>
                  ) : (
                    <span className="text-mist/50">—</span>
                  )}
                </td>
                <td className={`${td} whitespace-nowrap text-mist`}>{fmtDate(d.updated_at)}</td>
                <td className={td}>
                  <ReviewActions targetType="object_draft" targetId={d.id} actions={["feature", "hide"]} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section className="mt-8 pb-6">
        <div className="zx-card p-4">
          <p className="text-sm text-porcelain">Curation, not commerce</p>
          <p className="mt-2 text-xs leading-relaxed text-mist">
            <span className="text-gold">Feature</span> surfaces a public draft on inspiration surfaces
            across the platform; <span className="text-porcelain">Hide</span> removes it from public
            display without touching the owner's private archive. Drafts never enter a cart from this
            console — any physical outcome starts with the owner and passes through human review, quotation,
            and compliance checks in Trade and Orders.
          </p>
        </div>
      </section>
    </div>
  );
}
