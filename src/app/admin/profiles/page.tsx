import { Notice, Tag } from "@/components/ui";
import { db } from "@/lib/store";
import type { UserProfile } from "@/lib/types";

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

const privacyTone: Record<UserProfile["privacy_level"], "default" | "gold" | "jade"> = {
  private: "default",
  "co-create": "gold",
  public: "jade",
};

function chipGroups(p: UserProfile): { label: string; items: string[] }[] {
  return [
    { label: "Scent preferences", items: p.scent_preferences ?? [] },
    { label: "Alcohol preferences", items: p.alcohol_preferences ?? [] },
    { label: "Favorite colors", items: p.favorite_colors ?? [] },
    { label: "Visual language", items: p.visual_preferences ?? [] },
    { label: "Gifting stance", items: p.gift_preferences ?? [] },
    { label: "Personality tags", items: p.personality_tags ?? [] },
    { label: "Common scenarios", items: p.common_scenarios ?? [] },
  ].filter((g) => g.items.length > 0);
}

function factRows(p: UserProfile): { label: string; value: string }[] {
  const rows: { label: string; value?: string }[] = [
    { label: "Age range", value: p.age_range },
    { label: "Address style", value: p.address_style },
    { label: "Budget range", value: p.budget_range },
    { label: "Emotional state", value: p.emotional_state },
    {
      label: "Alcohol tolerance",
      value: p.alcohol_tolerance
        ? `${p.alcohol_tolerance}${p.non_alcohol_ok ? " · zero-proof welcome" : ""}`
        : undefined,
    },
    { label: "Music", value: p.music },
    { label: "Movies", value: p.movies },
    { label: "Cities", value: p.cities },
    { label: "Literary imagery", value: p.literary_imagery },
  ];
  return rows.filter((r): r is { label: string; value: string } => Boolean(r.value));
}

export default async function AdminProfilesPage() {
  const data = db();
  const userById = new Map(data.users.map((u) => [u.id, u]));
  const profiles = data.user_profiles;
  const relationships = data.relationship_profiles;

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">User Profiles</h1>
      <p className="mt-1 text-sm text-mist">
        Preference profiles the concierge reads before proposing — {profiles.length} profiles and{" "}
        {relationships.length} relationship profiles on record.
      </p>

      <div className="mt-5">
        <Notice tone="gold" title="Self-expression only">
          MBTI, zodiac, and blood type are optional style signals that shape tone and imagery in AI
          proposals. They never influence access, pricing, or eligibility, are excluded from public display
          by default, and each member can edit or delete them at any time.
        </Notice>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        {profiles.map((p) => {
          const owner = userById.get(p.user_id);
          const selfTags = [
            p.mbti ? `MBTI · ${p.mbti}` : null,
            p.zodiac ? `Zodiac · ${p.zodiac}` : null,
            p.blood_type ? `Blood · ${p.blood_type}` : null,
          ].filter((t): t is string => t !== null);
          return (
            <div key={p.id} className="zx-card p-5">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="font-display text-base text-porcelain">
                    {owner?.nickname ?? p.user_id}
                    {p.nickname ? <span className="text-mist"> · goes by “{p.nickname}”</span> : null}
                  </p>
                  <p className="mt-0.5 text-xs text-mist">{owner?.email ?? "Account record"} · updated {fmtDate(p.updated_at)}</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Tag tone={privacyTone[p.privacy_level]}>{p.privacy_level}</Tag>
                  <Tag tone={p.memory_enabled ? "jade" : "default"}>
                    {p.memory_enabled ? "memory on" : "memory off"}
                  </Tag>
                </div>
              </div>

              {selfTags.length > 0 && (
                <div className="mt-4">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-mist">
                    Self-expression
                  </p>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {selfTags.map((t) => (
                      <Tag key={t} tone="gold">
                        {t}
                      </Tag>
                    ))}
                  </div>
                </div>
              )}

              <div className="mt-4 space-y-3">
                {chipGroups(p).map((g) => (
                  <div key={g.label}>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-mist">{g.label}</p>
                    <div className="mt-1.5 flex flex-wrap gap-1.5">
                      {g.items.map((item) => (
                        <Tag key={item}>{item}</Tag>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              <dl className="mt-4 border-t border-hairline pt-1">
                {factRows(p).map((r) => (
                  <div key={r.label} className="flex gap-3 border-b border-hairline/50 py-2 last:border-0">
                    <dt className="w-32 shrink-0 text-xs uppercase tracking-wider text-mist">{r.label}</dt>
                    <dd className="text-sm text-porcelain">{r.value}</dd>
                  </div>
                ))}
              </dl>
            </div>
          );
        })}
      </div>

      <section className="mt-10 pb-6">
        <h2 className="font-display text-lg text-porcelain">Relationship profiles</h2>
        <p className="mt-1 text-xs text-mist">
          Recipients that members keep on file so gift proposals arrive pre-tuned — private by default,
          visible only to their owner and this console.
        </p>
        <div className="zx-card mt-3 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <th className={th}>Owner</th>
                <th className={th}>Relation</th>
                <th className={th}>Nickname</th>
                <th className={th}>Age range</th>
                <th className={th}>Preferences</th>
                <th className={th}>Important dates</th>
                <th className={th}>Privacy</th>
                <th className={th}>Updated</th>
              </tr>
            </thead>
            <tbody>
              {relationships.map((r) => (
                <tr key={r.id} className="border-b border-hairline/50 last:border-0">
                  <td className={`${td} whitespace-nowrap text-porcelain`}>
                    {userById.get(r.user_id)?.nickname ?? r.user_id}
                  </td>
                  <td className={`${td} whitespace-nowrap text-mist`}>{r.relation_type}</td>
                  <td className={`${td} whitespace-nowrap text-porcelain`}>{r.nickname}</td>
                  <td className={`${td} whitespace-nowrap text-mist`}>{r.age_range ?? "—"}</td>
                  <td className={`${td} min-w-56 text-mist`}>{r.preferences ?? "—"}</td>
                  <td className={`${td} whitespace-nowrap text-mist`}>{r.important_dates ?? "—"}</td>
                  <td className={td}>
                    <Tag tone={r.privacy_level === "public" ? "jade" : "default"}>{r.privacy_level}</Tag>
                  </td>
                  <td className={`${td} whitespace-nowrap text-mist`}>{fmtDate(r.updated_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
