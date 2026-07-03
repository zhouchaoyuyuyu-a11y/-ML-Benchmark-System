"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button, Tag } from "@/components/ui";
import type { CmsBlock } from "@/lib/types";

const textareaCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none resize-y";

const consumers: Record<string, string> = {
  "home.hero.badge": "Rendered as the capsule badge above the homepage hero headline (/).",
  "home.announcement": "The announcement bar above the homepage hero, linking to the co-creation pool (/).",
  "maison.concierge.hours": "Shown in the human-concierge section of the Maison page (/maison).",
  "download.beta.note": "Distribution note above the install buttons on the download page (/download).",
};

type BlockStatus = "idle" | "saving" | "saved" | "error";

export default function CmsClient({ blocks }: { blocks: CmsBlock[] }) {
  const router = useRouter();
  const [drafts, setDrafts] = useState<Record<string, { content: string; enabled: boolean }>>(
    Object.fromEntries(blocks.map((b) => [b.id, { content: b.content, enabled: b.enabled }])),
  );
  const [status, setStatus] = useState<Record<string, BlockStatus>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  function update(id: string, patch: Partial<{ content: string; enabled: boolean }>) {
    setDrafts((d) => ({ ...d, [id]: { ...d[id], ...patch } }));
    setStatus((s) => ({ ...s, [id]: "idle" }));
  }

  async function saveBlock(id: string) {
    const draft = drafts[id];
    if (!draft) return;
    setStatus((s) => ({ ...s, [id]: "saving" }));
    setErrors((e) => ({ ...e, [id]: "" }));
    try {
      const res = await fetch("/api/admin/config", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          section: "cms",
          patch: { id, content: draft.content, enabled: draft.enabled },
        }),
      });
      const data = await res.json().catch(() => ({ ok: false }));
      if (!data.ok) {
        setStatus((s) => ({ ...s, [id]: "error" }));
        setErrors((e) => ({ ...e, [id]: data.error ?? "Save failed — try again." }));
      } else {
        setStatus((s) => ({ ...s, [id]: "saved" }));
        router.refresh();
      }
    } catch {
      setStatus((s) => ({ ...s, [id]: "error" }));
      setErrors((e) => ({ ...e, [id]: "Network error — the block was not saved." }));
    }
  }

  return (
    <div className="space-y-4">
      {blocks.map((block) => {
        const draft = drafts[block.id] ?? { content: block.content, enabled: block.enabled };
        const st = status[block.id] ?? "idle";
        return (
          <div key={block.id} className="zx-card p-5">
            <div className="flex flex-wrap items-center gap-3">
              <code className="rounded bg-veil px-2 py-1 font-mono text-xs text-gold">{block.key}</code>
              <Tag>{block.page}</Tag>
              {draft.enabled ? <Tag tone="jade">rendering</Tag> : <Tag tone="ember">hidden</Tag>}
            </div>
            <p className="mt-2 text-sm text-porcelain">{block.title}</p>
            <p className="mt-1 text-xs leading-relaxed text-mist">
              {consumers[block.key] ?? `Consumed by the ${block.page} page, looked up by key.`}
            </p>
            <textarea
              value={draft.content}
              onChange={(e) => update(block.id, { content: e.target.value })}
              rows={3}
              className={`${textareaCls} mt-3`}
              placeholder="Block content"
            />
            <div className="mt-3 flex flex-wrap items-center gap-4">
              <label className="flex cursor-pointer items-center gap-2.5 text-sm text-porcelain">
                <input
                  type="checkbox"
                  checked={draft.enabled}
                  onChange={(e) => update(block.id, { enabled: e.target.checked })}
                  className="h-4 w-4 rounded border-hairline bg-ink accent-gold"
                />
                Enabled
              </label>
              <Button
                variant="outline"
                className="!px-4 !py-2 !text-xs"
                onClick={() => saveBlock(block.id)}
                disabled={st === "saving"}
              >
                {st === "saving" ? "Saving…" : "Save block"}
              </Button>
              {st === "saved" && <span className="text-sm font-medium text-jade">Saved ✓</span>}
              {st === "error" && <span className="text-sm text-ember">{errors[block.id] || "Save failed."}</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}
