"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button, Tag } from "@/components/ui";
import type { SocialAccount } from "@/lib/types";

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

interface RowState {
  id: string;
  platform: string;
  icon: string;
  official_url: string;
  tracking_params: string;
  backup_url: string;
  display_order: string;
  enabled: boolean;
}

type RowStatus = "idle" | "saving" | "saved" | "error";

export default function SocialConfigClient({ accounts }: { accounts: SocialAccount[] }) {
  const router = useRouter();
  const [rows, setRows] = useState<RowState[]>(
    accounts.map((a) => ({
      id: a.id,
      platform: a.platform,
      icon: a.icon,
      official_url: a.official_url,
      tracking_params: a.tracking_params ?? "",
      backup_url: a.backup_url ?? "",
      display_order: String(a.display_order),
      enabled: a.enabled,
    })),
  );
  const [status, setStatus] = useState<Record<string, RowStatus>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  function update(id: string, patch: Partial<RowState>) {
    setRows((rs) => rs.map((r) => (r.id === id ? { ...r, ...patch } : r)));
    setStatus((s) => ({ ...s, [id]: "idle" }));
  }

  async function saveRow(id: string) {
    const row = rows.find((r) => r.id === id);
    if (!row) return;
    const order = Number(row.display_order);
    if (!Number.isFinite(order) || order < 0) {
      setStatus((s) => ({ ...s, [id]: "error" }));
      setErrors((e) => ({ ...e, [id]: "Display order must be a non-negative number." }));
      return;
    }
    setStatus((s) => ({ ...s, [id]: "saving" }));
    setErrors((e) => ({ ...e, [id]: "" }));
    try {
      const res = await fetch("/api/admin/config", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          section: "social",
          patch: {
            id,
            official_url: row.official_url,
            tracking_params: row.tracking_params,
            backup_url: row.backup_url,
            display_order: order,
            enabled: row.enabled,
          },
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
      setErrors((e) => ({ ...e, [id]: "Network error — the row was not saved." }));
    }
  }

  return (
    <div className="space-y-4">
      {rows.map((row) => {
        const st = status[row.id] ?? "idle";
        return (
          <div key={row.id} className="zx-card p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <p className="font-display text-base text-porcelain">{row.platform}</p>
                <span className="font-mono text-[10px] text-mist/70">icon: {row.icon}</span>
              </div>
              {row.enabled ? <Tag tone="jade">visible</Tag> : <Tag>hidden</Tag>}
            </div>

            <div className="mt-4 grid gap-4 lg:grid-cols-2">
              <label className="block">
                <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Official URL</span>
                <input
                  value={row.official_url}
                  onChange={(e) => update(row.id, { official_url: e.target.value })}
                  className={inputCls}
                  placeholder="https://…"
                />
              </label>
              <label className="block">
                <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Tracking parameters</span>
                <input
                  value={row.tracking_params}
                  onChange={(e) => update(row.id, { tracking_params: e.target.value })}
                  className={`${inputCls} font-mono`}
                  placeholder="utm_source=…"
                />
              </label>
              <label className="block">
                <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Backup URL</span>
                <input
                  value={row.backup_url}
                  onChange={(e) => update(row.id, { backup_url: e.target.value })}
                  className={inputCls}
                  placeholder="Mirror or regional link (optional)"
                />
              </label>
              <div className="grid grid-cols-2 items-end gap-4">
                <label className="block">
                  <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Display order</span>
                  <input
                    type="number"
                    min={0}
                    value={row.display_order}
                    onChange={(e) => update(row.id, { display_order: e.target.value })}
                    className={inputCls}
                  />
                </label>
                <label className="flex cursor-pointer items-center gap-2.5 pb-2.5 text-sm text-porcelain">
                  <input
                    type="checkbox"
                    checked={row.enabled}
                    onChange={(e) => update(row.id, { enabled: e.target.checked })}
                    className="h-4 w-4 rounded border-hairline bg-ink accent-gold"
                  />
                  Enabled
                </label>
              </div>
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-3">
              <Button
                variant="outline"
                className="!px-4 !py-2 !text-xs"
                onClick={() => saveRow(row.id)}
                disabled={st === "saving"}
              >
                {st === "saving" ? "Saving…" : `Save ${row.platform}`}
              </Button>
              {st === "saved" && <span className="text-sm font-medium text-jade">Saved ✓</span>}
              {st === "error" && <span className="text-sm text-ember">{errors[row.id] || "Save failed."}</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}
