import { Tag } from "@/components/ui";
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

export default async function AdminDesignsPage() {
  const data = db();
  const draftById = new Map(data.object_drafts.map((d) => [d.id, d]));
  const versions = [...data.design_versions].sort((a, b) => b.created_at.localeCompare(a.created_at));
  const latest = versions[0];
  const latestDraft = latest ? draftById.get(latest.object_draft_id) : undefined;

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">Design Versions</h1>
      <p className="mt-1 text-sm text-mist">
        Immutable snapshots of bottle, label, packaging, and liquid decisions — {versions.length} versions
        across {new Set(versions.map((v) => v.object_draft_id)).size} drafts. Each version carries a
        content-derived hash that certificates and casting quotes reference.
      </p>

      <div className="zx-card mt-6 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className={th}>Version</th>
              <th className={th}>Draft</th>
              <th className={th}>Version hash</th>
              <th className={th}>Payload keys</th>
              <th className={th}>Created</th>
            </tr>
          </thead>
          <tbody>
            {versions.map((v) => {
              const draft = draftById.get(v.object_draft_id);
              return (
                <tr key={v.id} className="border-b border-hairline/50 last:border-0">
                  <td className={`${td} whitespace-nowrap`}>
                    <p className="text-porcelain">{v.version_name}</p>
                    <p className="font-mono text-[10px] text-mist/70">{v.id}</p>
                  </td>
                  <td className={`${td} whitespace-nowrap`}>
                    <p className="text-porcelain">{draft?.title ?? v.object_draft_id}</p>
                    {draft && (
                      <p className="text-[10px] text-mist/70">{draft.object_type.replace(/_/g, " ")}</p>
                    )}
                  </td>
                  <td className={`${td} whitespace-nowrap`}>
                    <code className="rounded border border-hairline bg-ink px-1.5 py-0.5 font-mono text-xs text-gold">
                      {v.version_hash}
                    </code>
                  </td>
                  <td className={td}>
                    <div className="flex max-w-64 flex-wrap gap-1">
                      {Object.keys(v.design_payload).map((k) => (
                        <Tag key={k}>{k}</Tag>
                      ))}
                    </div>
                  </td>
                  <td className={`${td} whitespace-nowrap text-mist`}>{fmtDate(v.created_at)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <section className="mt-10">
        <h2 className="font-display text-lg text-porcelain">How version hashing works</h2>
        <p className="mt-1 text-xs text-mist">
          The rules that keep every archived design trustworthy — from studio preview to Reserve certificate.
        </p>
        <div className="mt-3 grid gap-3 lg:grid-cols-3">
          <div className="zx-card p-4">
            <p className="text-sm text-porcelain">Content-derived fingerprint</p>
            <p className="mt-2 text-xs leading-relaxed text-mist">
              Each <code className="font-mono text-gold">zx-</code> hash is computed deterministically from
              the full design payload — bottle, label, packaging, and liquid fields together. The same
              payload always produces the same hash; changing a single character produces a different one.
            </p>
          </div>
          <div className="zx-card p-4">
            <p className="text-sm text-porcelain">Versions are immutable</p>
            <p className="mt-2 text-xs leading-relaxed text-mist">
              A version is never edited in place. Refinements create a new version with its own name and
              hash, so the lineage of a design — v1, v2, and onward — stays fully auditable. If a payload
              were altered, its stored hash would no longer match a recomputation, which is how tampering
              is detected.
            </p>
          </div>
          <div className="zx-card p-4">
            <p className="text-sm text-porcelain">Downstream references</p>
            <p className="mt-2 text-xs leading-relaxed text-mist">
              Reserve records bind to a specific version id, casting quotes price a specific hash, and QR
              certificates display it. Immutability guarantees that what a customer scans is exactly what
              was reviewed, quoted, and produced.
            </p>
          </div>
        </div>
      </section>

      {latest && (
        <section className="mt-10 pb-6">
          <h2 className="font-display text-lg text-porcelain">Latest payload inspected</h2>
          <p className="mt-1 text-xs text-mist">
            {latest.version_name}
            {latestDraft ? ` · ${latestDraft.title}` : ""} ·{" "}
            <code className="font-mono text-gold">{latest.version_hash}</code>
          </p>
          <div className="zx-card mt-3 p-4">
            <dl>
              {Object.entries(latest.design_payload).map(([k, v]) => (
                <div
                  key={k}
                  className="grid grid-cols-1 gap-1 border-b border-hairline/50 py-2.5 last:border-0 sm:grid-cols-[140px_1fr] sm:gap-4"
                >
                  <dt className="text-xs uppercase tracking-wider text-mist">{k}</dt>
                  <dd className="text-sm text-porcelain">{v}</dd>
                </div>
              ))}
            </dl>
          </div>
        </section>
      )}
    </div>
  );
}
