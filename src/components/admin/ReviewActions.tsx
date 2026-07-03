"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

/** Admin review action buttons — posts to /api/admin/review and refreshes. */
export default function ReviewActions({
  targetType,
  targetId,
  actions = ["approve", "reject", "revision", "escalate"],
}: {
  targetType: "co_creation_project" | "trade_request" | "moderation_log" | "object_draft" | "content_calendar";
  targetId: string;
  actions?: string[];
}) {
  const router = useRouter();
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function act(action: string) {
    setBusy(action);
    setError(null);
    const res = await fetch("/api/admin/review", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ targetType, targetId, action }),
    });
    const data = await res.json().catch(() => ({ ok: false }));
    setBusy(null);
    if (!data.ok) {
      setError(data.error ?? "Action failed");
      return;
    }
    router.refresh();
  }

  const labels: Record<string, { text: string; cls: string }> = {
    approve: { text: "Approve", cls: "border-jade/50 text-jade hover:bg-jade/10" },
    reject: { text: "Reject", cls: "border-ember/50 text-ember hover:bg-ember/10" },
    revision: { text: "Request revision", cls: "border-gold/50 text-gold hover:bg-gold/10" },
    escalate: { text: "To concierge", cls: "border-supply/50 text-supply hover:bg-supply/10" },
    feature: { text: "Feature", cls: "border-gold/50 text-gold hover:bg-gold/10" },
    hide: { text: "Hide", cls: "border-hairline text-mist hover:text-porcelain" },
    unpublish: { text: "Unpublish", cls: "border-ember/50 text-ember hover:bg-ember/10" },
    compliance_risk: { text: "Mark compliance risk", cls: "border-ember/50 text-ember hover:bg-ember/10" },
    infeasible: { text: "Supply-chain infeasible", cls: "border-ember/50 text-ember hover:bg-ember/10" },
  };

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {actions.map((a) => (
        <button
          key={a}
          onClick={() => act(a)}
          disabled={busy !== null}
          className={`rounded-md border px-2 py-1 text-[11px] transition-colors disabled:opacity-40 ${labels[a]?.cls ?? "border-hairline text-mist"}`}
        >
          {busy === a ? "…" : labels[a]?.text ?? a}
        </button>
      ))}
      {error && <span className="text-[11px] text-ember">{error}</span>}
    </div>
  );
}
