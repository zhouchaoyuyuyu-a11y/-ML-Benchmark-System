import Link from "next/link";
import { Stat, StatusPill, Tag } from "@/components/ui";
import { db } from "@/lib/store";

const th = "border-b border-hairline px-3 py-2 text-left text-xs uppercase tracking-wider text-mist";
const td = "px-3 py-2.5";

const typeTone: Record<string, "default" | "gold" | "supply" | "jade" | "ember"> = {
  spirit: "gold",
  fragrance: "jade",
  bottle: "default",
  giftbox: "supply",
  label: "supply",
  enterprise_gift: "gold",
  badge: "supply",
  co_creation: "jade",
  design_version: "default",
};

export default async function AdminReservePage() {
  const data = db();
  const userById = new Map(data.users.map((u) => [u.id, u]));
  const records = [...data.reserve_records].sort((a, b) => b.created_at.localeCompare(a.created_at));

  const publicCount = records.filter((r) => r.privacy_level === "public").length;
  const inProduction = records.filter((r) => r.delivery_status === "in_production").length;
  const aftercareActive = records.filter((r) => r.aftercare_status === "active").length;

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">Reserve Records</h1>
      <p className="mt-1 text-sm text-mist">
        The archive of object identities — {records.length} records, each with a ZOTAIX ID, a QR/NFC
        binding, and a certificate page. The record outlasts the bottle: aftercare and replenishment attach
        here, not to the physical object.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Records" value={String(records.length)} hint="All archived identities" />
        <Stat label="Public" value={String(publicCount)} hint="Scannable by anyone" />
        <Stat label="In production" value={String(inProduction)} hint="Physical path underway" />
        <Stat label="Aftercare active" value={String(aftercareActive)} hint="Care window open" />
      </div>

      <div className="zx-card mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className={th}>ZOTAIX ID</th>
              <th className={th}>Object</th>
              <th className={th}>Owner</th>
              <th className={th}>Type</th>
              <th className={th}>Privacy</th>
              <th className={th}>Delivery</th>
              <th className={th}>Aftercare</th>
              <th className={th}>QR / NFC</th>
            </tr>
          </thead>
          <tbody>
            {records.map((r) => (
              <tr key={r.id} className="border-b border-hairline/50 last:border-0">
                <td className={`${td} whitespace-nowrap`}>
                  <Link href={`/reserve/${r.id}`} className="font-mono text-xs text-gold hover:underline">
                    {r.zotaix_id}
                  </Link>
                </td>
                <td className={`${td} whitespace-nowrap`}>
                  <p className="text-porcelain">{r.object_name}</p>
                  {r.relationship_scene && (
                    <p className="text-[10px] text-mist/70">{r.relationship_scene}</p>
                  )}
                </td>
                <td className={`${td} whitespace-nowrap text-mist`}>
                  {userById.get(r.user_id)?.nickname ?? r.user_id}
                </td>
                <td className={td}>
                  <Tag tone={typeTone[r.object_type] ?? "default"}>
                    {r.object_type.replace(/_/g, " ")}
                  </Tag>
                </td>
                <td className={td}>
                  <Tag tone={r.privacy_level === "public" ? "jade" : "default"}>{r.privacy_level}</Tag>
                </td>
                <td className={td}>
                  <StatusPill status={r.delivery_status} />
                </td>
                <td className={td}>
                  <StatusPill status={r.aftercare_status} />
                </td>
                <td className={`${td} whitespace-nowrap font-mono text-xs text-mist`}>{r.qr_nfc_id}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section className="mt-8">
        <div className="zx-card p-4">
          <p className="text-sm text-porcelain">Certificates</p>
          <p className="mt-2 text-xs leading-relaxed text-mist">
            Every record renders a certificate page at{" "}
            <code className="font-mono text-gold">/reserve/[id]</code> — the page a QR or NFC scan resolves
            to. Public records open for anyone who scans; private records stay sealed to the owner's
            session. The certificate shows the object's story, emotion tags, bound design version, and —
            where a physical path exists — batch and delivery lineage.{" "}
            <Link href="/reserve" className="text-gold hover:underline">
              Open the public archive →
            </Link>
          </p>
        </div>
      </section>

      <section className="mt-8 pb-6">
        <h2 className="font-display text-lg text-porcelain">Aftercare states</h2>
        <p className="mt-1 text-xs text-mist">
          Aftercare belongs to the record, which is how an anniversary bottle becomes an annual ritual.
        </p>
        <div className="mt-3 grid gap-3 sm:grid-cols-3">
          <div className="zx-card p-4">
            <Tag tone="jade">active</Tag>
            <p className="mt-2 text-xs leading-relaxed text-mist">
              The care window is open: replenishment requests, engraving refresh, and concierge follow-up
              are all available from the certificate page.
            </p>
          </div>
          <div className="zx-card p-4">
            <Tag tone="gold">expired</Tag>
            <p className="mt-2 text-xs leading-relaxed text-mist">
              The window has closed. The archive stays intact, and the owner can reopen aftercare through
              a human concierge request at any time.
            </p>
          </div>
          <div className="zx-card p-4">
            <Tag>none</Tag>
            <p className="mt-2 text-xs leading-relaxed text-mist">
              A digital-only identity — badges, marks, and directions kept without a physical delivery. No
              care window applies until a physical path begins.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
