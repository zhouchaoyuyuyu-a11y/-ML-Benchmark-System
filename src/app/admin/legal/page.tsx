import Link from "next/link";
import { Notice, Stat, Tag } from "@/components/ui";
import { complianceNotice, profileNotice } from "@/lib/copy";
import { db } from "@/lib/store";

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

export default async function AdminLegalPage() {
  const data = db();
  const docs = [...data.legal_docs].sort((a, b) => a.slug.localeCompare(b.slug));
  const latestEffective = docs.reduce((max, d) => (d.effective_date > max ? d.effective_date : max), "");

  const inventory: { notice: string; text: string; surfaces: string; doc: string }[] = [
    {
      notice: "Alcohol & AI creative-proposal notice",
      text: complianceNotice.en,
      surfaces:
        "Rendered as a gold Notice on the homepage, Forge, Maison, Trade, market detail, and co-creation pages — every flow that can lead to a physical alcohol outcome repeats it before the human-review step.",
      doc: "/legal/alcohol · /legal/ai",
    },
    {
      notice: "Age gate",
      text: "A confirm-your-age overlay on alcohol-related pages, with region compliance mentioned alongside it.",
      surfaces:
        "The AgeGate component covers Maison, Forge, Trade, and market detail pages; the age_gate_enabled switch in Global Settings controls it platform-wide.",
      doc: "/legal/alcohol · /legal/minors",
    },
    {
      notice: "AI-generated content marking",
      text: "Structured concierge results are presented as creative proposals, with model and fallback labeling on every generation.",
      surfaces:
        "ProposalCard output across Concierge, Forge, Supply, and Maison; the full policy lives on the AI Generated Content Notice page.",
      doc: "/legal/ai",
    },
    {
      notice: "Profile self-expression notice",
      text: profileNotice.en,
      surfaces:
        "Shown wherever MBTI, zodiac, blood type, or preference tags are collected or edited — the profile center and onboarding preference forms.",
      doc: "/legal/privacy",
    },
    {
      notice: "Minor protection",
      text: "Zero-proof framing rules, no alcohol imagery adjacent to minor-oriented content, and a dedicated minor_safety moderation dimension.",
      surfaces:
        "Enforced through the moderation queue on every public listing and co-creation project; stated on the Minor Protection Notice page.",
      doc: "/legal/minors",
    },
  ];

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">Legal Pages</h1>
      <p className="mt-1 text-sm text-mist">
        The registry of the platform's {docs.length} legal and compliance documents — versions, effective
        dates, and the public routes where each one is served.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Documents" value={String(docs.length)} hint="Public legal pages" />
        <Stat label="Latest effective" value={latestEffective || "—"} hint="Most recent effective date" />
        <Stat label="Compliance notices" value={String(inventory.length)} hint="Recurring on-page notices" />
        <Stat label="Serving route" value="/legal/*" hint="Linked from every footer" />
      </div>

      <div className="zx-card mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className={th}>Document</th>
              <th className={th}>Public page</th>
              <th className={th}>Version</th>
              <th className={th}>Effective</th>
              <th className={th}>Updated</th>
            </tr>
          </thead>
          <tbody>
            {docs.map((doc) => (
              <tr key={doc.id} className="border-b border-hairline/50 last:border-0">
                <td className={`${td} whitespace-nowrap text-porcelain`}>{doc.title}</td>
                <td className={`${td} whitespace-nowrap`}>
                  <Link href={`/legal/${doc.slug}`} className="font-mono text-xs text-gold hover:underline">
                    /legal/{doc.slug} →
                  </Link>
                </td>
                <td className={td}>
                  <Tag tone="gold">v{doc.version}</Tag>
                </td>
                <td className={`${td} whitespace-nowrap text-mist`}>{doc.effective_date}</td>
                <td className={`${td} whitespace-nowrap text-mist`}>{fmtDate(doc.updated_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-6">
        <Notice tone="gold" title="Editorial workflow">
          Legal text lives in the codebase under the /legal routes so every change ships through code review
          and a deployment — a deliberate, review-controlled path for language with legal weight. This registry
          tracks the resulting versions and effective dates; when counsel approves new wording, the page ships
          and the version and effective date are advanced together here.
        </Notice>
      </div>

      <section className="mt-10 pb-6">
        <h2 className="font-display text-lg text-porcelain">Compliance notice inventory</h2>
        <p className="mt-1 text-xs text-mist">
          The recurring notices woven into product surfaces, the exact language they carry, and the documents
          that govern them.
        </p>
        <div className="mt-3 space-y-3">
          {inventory.map((n) => (
            <div key={n.notice} className="zx-card p-5">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="text-sm text-porcelain">{n.notice}</p>
                <span className="font-mono text-[10px] text-gold">{n.doc}</span>
              </div>
              <p className="mt-2 border-l-2 border-gold/40 pl-3 text-xs italic leading-relaxed text-mist">
                “{n.text}”
              </p>
              <p className="mt-2 text-xs leading-relaxed text-mist">{n.surfaces}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
