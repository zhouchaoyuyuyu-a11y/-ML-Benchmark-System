"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button, StatusPill } from "@/components/ui";
import type { ContentCalendarItem } from "@/lib/types";

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";
const th = "border-b border-hairline px-3 py-2 text-left text-xs uppercase tracking-wider text-mist";
const td = "px-3 py-2.5 align-top";

const platforms = ["WeChat", "Instagram", "TikTok", "X", "YouTube", "LinkedIn"];
const statuses: ContentCalendarItem["status"][] = ["draft", "scheduled", "published"];

function fmtSchedule(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "UTC",
  });
}

export default function CalendarClient({ items }: { items: ContentCalendarItem[] }) {
  const router = useRouter();
  const [busyId, setBusyId] = useState<string | null>(null);
  const [rowError, setRowError] = useState<Record<string, string>>({});

  const [platform, setPlatform] = useState(platforms[0]);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [scheduledAt, setScheduledAt] = useState("");
  const [owner, setOwner] = useState("");
  const [relatedUrl, setRelatedUrl] = useState("");
  const [creating, setCreating] = useState(false);
  const [created, setCreated] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  async function setStatus(id: string, status: ContentCalendarItem["status"]) {
    setBusyId(id);
    setRowError((e) => ({ ...e, [id]: "" }));
    try {
      const res = await fetch("/api/admin/config", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ section: "calendar", patch: { id, status } }),
      });
      const data = await res.json().catch(() => ({ ok: false }));
      if (!data.ok) {
        setRowError((e) => ({ ...e, [id]: data.error ?? "Update failed." }));
      } else {
        router.refresh();
      }
    } catch {
      setRowError((e) => ({ ...e, [id]: "Network error — status unchanged." }));
    } finally {
      setBusyId(null);
    }
  }

  async function createItem(e: React.FormEvent) {
    e.preventDefault();
    setCreated(false);
    setCreateError(null);
    if (!title.trim() || !scheduledAt) {
      setCreateError("Title and scheduled time are required.");
      return;
    }
    setCreating(true);
    try {
      const res = await fetch("/api/admin/config", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          section: "calendar",
          patch: {
            platform,
            title: title.trim(),
            content: content.trim(),
            scheduled_at: new Date(scheduledAt).toISOString(),
            owner: owner.trim() || undefined,
            related_url: relatedUrl.trim(),
          },
        }),
      });
      const data = await res.json().catch(() => ({ ok: false }));
      if (!data.ok) {
        setCreateError(data.error ?? "Item was not created — try again.");
      } else {
        setCreated(true);
        setTitle("");
        setContent("");
        setScheduledAt("");
        setRelatedUrl("");
        router.refresh();
      }
    } catch {
      setCreateError("Network error — the item was not created.");
    } finally {
      setCreating(false);
    }
  }

  return (
    <div className="space-y-8">
      <div className="zx-card overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className={th}>Scheduled</th>
              <th className={th}>Platform</th>
              <th className={th}>Item</th>
              <th className={th}>Owner</th>
              <th className={th}>Link</th>
              <th className={th}>Status</th>
              <th className={th}>Move to</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.id} className="border-b border-hairline/50 last:border-0">
                <td className={`${td} whitespace-nowrap text-mist`}>{fmtSchedule(item.scheduled_at)}</td>
                <td className={`${td} whitespace-nowrap text-porcelain`}>{item.platform}</td>
                <td className={td}>
                  <p className="min-w-48 max-w-72 text-porcelain">{item.title}</p>
                  <p className="mt-0.5 line-clamp-2 max-w-72 text-xs leading-relaxed text-mist">{item.content}</p>
                </td>
                <td className={`${td} whitespace-nowrap text-mist`}>{item.owner}</td>
                <td className={`${td} whitespace-nowrap`}>
                  {item.related_url ? (
                    <span className="font-mono text-xs text-gold">{item.related_url}</span>
                  ) : (
                    <span className="text-xs text-mist/60">—</span>
                  )}
                </td>
                <td className={td}>
                  <StatusPill status={item.status} />
                </td>
                <td className={td}>
                  <div className="flex flex-wrap items-center gap-1.5">
                    {statuses
                      .filter((s) => s !== item.status)
                      .map((s) => (
                        <button
                          key={s}
                          onClick={() => setStatus(item.id, s)}
                          disabled={busyId !== null}
                          className="rounded-md border border-hairline px-2 py-1 text-[11px] text-mist transition-colors hover:border-gold hover:text-gold disabled:opacity-40"
                        >
                          {busyId === item.id ? "…" : s}
                        </button>
                      ))}
                  </div>
                  {rowError[item.id] && <p className="mt-1 text-[11px] text-ember">{rowError[item.id]}</p>}
                </td>
              </tr>
            ))}
            {items.length === 0 && (
              <tr>
                <td className={`${td} text-mist`} colSpan={7}>
                  The pipeline is empty — add the first item below.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <form onSubmit={createItem} className="zx-card p-5">
        <h2 className="font-display text-lg text-porcelain">New item</h2>
        <p className="mt-1 text-xs text-mist">
          New items enter the pipeline as drafts; move them to scheduled once the copy and assets are locked.
        </p>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <label className="block">
            <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Platform</span>
            <select value={platform} onChange={(e) => setPlatform(e.target.value)} className={inputCls}>
              {platforms.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Title</span>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className={inputCls}
              placeholder="Post working title"
            />
          </label>
          <label className="block">
            <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Scheduled at</span>
            <input
              type="datetime-local"
              value={scheduledAt}
              onChange={(e) => setScheduledAt(e.target.value)}
              className={inputCls}
            />
          </label>
          <label className="block sm:col-span-2 lg:col-span-3">
            <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Content</span>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              rows={3}
              className={`${inputCls} resize-y`}
              placeholder="Post copy, hook, or brief for the owner"
            />
          </label>
          <label className="block">
            <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Owner</span>
            <input
              value={owner}
              onChange={(e) => setOwner(e.target.value)}
              className={inputCls}
              placeholder="Team · Name"
            />
          </label>
          <label className="block sm:col-span-2">
            <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Related URL</span>
            <input
              value={relatedUrl}
              onChange={(e) => setRelatedUrl(e.target.value)}
              className={inputCls}
              placeholder="/co-create or a campaign landing path"
            />
          </label>
        </div>
        <div className="mt-5 flex flex-wrap items-center gap-3">
          <Button type="submit" disabled={creating}>
            {creating ? "Adding…" : "Add to calendar"}
          </Button>
          {created && <span className="text-sm font-medium text-jade">Saved ✓ — added as draft</span>}
          {createError && <span className="text-sm text-ember">{createError}</span>}
        </div>
      </form>
    </div>
  );
}
