"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import type { ConceptProposal, NextActionKey } from "@/lib/types";

const ACTION_LABELS: Record<NextActionKey, { en: string; zh: string; href?: string }> = {
  save_inspiration: { en: "Save Inspiration", zh: "保存灵感" },
  emotional_card: { en: "Generate Emotional Card", zh: "生成情绪卡片" },
  label_copy: { en: "Generate Label Copy", zh: "生成瓶身文案" },
  add_preferences: { en: "Add to My Preferences", zh: "加入我的偏好", href: "/profile" },
  gift_draft: { en: "Create Gift Draft", zh: "创建礼物草案" },
  share: { en: "Share with Friends", zh: "分享给朋友" },
  co_create: { en: "Enter Co-Creation Pool", zh: "进入共创池", href: "/co-create" },
  physical_casting: { en: "Enter Physical Casting", zh: "进入实体铸造", href: "/trade" },
  human_concierge: { en: "Request Human Concierge", zh: "请求人工礼宾", href: "/maison#concierge" },
};

export default function ProposalCard({
  proposal,
  zh = false,
  compact = false,
}: {
  proposal: ConceptProposal;
  zh?: boolean;
  compact?: boolean;
}) {
  const router = useRouter();
  const [saving, setSaving] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [cardUrl, setCardUrl] = useState<string | null>(null);

  async function handleAction(key: NextActionKey) {
    const def = ACTION_LABELS[key];
    if (key === "share") {
      const text = `${proposal.label_copy ?? proposal.emotional_signal} — ZOTAIX`;
      const url = typeof window !== "undefined" ? window.location.href : "";
      if (navigator.share) {
        try {
          await navigator.share({ title: "ZOTAIX", text, url });
        } catch {
          /* user dismissed */
        }
      } else {
        await navigator.clipboard.writeText(`${text} ${url}`);
        setSaved("share");
        setTimeout(() => setSaved(null), 2000);
      }
      return;
    }
    if (key === "emotional_card") {
      const params = new URLSearchParams({
        copy: proposal.label_copy ?? proposal.emotional_signal,
        mark: proposal.digital_mark ?? "ZOTAIX Mark",
        keywords: proposal.keywords.join(" · "),
      });
      setCardUrl(`/api/card?${params.toString()}`);
      return;
    }
    if (key === "save_inspiration" || key === "gift_draft" || key === "label_copy") {
      setSaving(key);
      try {
        const res = await fetch("/api/drafts", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ proposal, intent: key }),
        });
        const data = await res.json();
        if (res.status === 401) {
          router.push("/login?next=/concierge");
          return;
        }
        if (data.ok) {
          setSaved(key);
          setTimeout(() => setSaved(null), 2500);
        }
      } finally {
        setSaving(null);
      }
      return;
    }
    if (def.href) router.push(def.href);
  }

  const rows: { label: string; labelZh: string; value?: string }[] = [
    { label: "Emotional signal", labelZh: "情绪信号", value: proposal.emotional_signal },
    { label: "Liquid direction", labelZh: "酒体方向", value: proposal.liquid_direction },
    { label: "Fragrance direction", labelZh: "香氛方向", value: proposal.scent_direction },
    { label: "Bottle & label direction", labelZh: "瓶身与标签方向", value: proposal.bottle_direction },
    { label: "Light suggestion", labelZh: "轻建议", value: proposal.suggestion },
  ];

  return (
    <div className="zx-card zx-fade-up overflow-hidden">
      <div className="border-b border-hairline bg-veil/40 px-5 py-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
            {proposal.kind === "daily" ? (zh ? "今日回应" : "Daily Response") : zh ? "结构化提案" : "Structured Proposal"}
          </span>
          {proposal.keywords.map((k) => (
            <span key={k} className="rounded-full border border-hairline px-2 py-0.5 text-xs text-mist">
              {k}
            </span>
          ))}
        </div>
      </div>

      <div className="space-y-3 px-5 py-4">
        {rows
          .filter((r) => r.value)
          .map((r) => (
            <div key={r.label}>
              <p className="text-xs uppercase tracking-wider text-mist">{zh ? r.labelZh : r.label}</p>
              <p className="mt-0.5 text-sm leading-relaxed text-porcelain">{r.value}</p>
            </div>
          ))}

        {proposal.names && proposal.names.length > 0 && (
          <div>
            <p className="text-xs uppercase tracking-wider text-mist">{zh ? "候选命名" : "Proposed names"}</p>
            <div className="mt-1.5 flex flex-wrap gap-2">
              {proposal.names.map((n) => (
                <span key={n} className="rounded-md border border-gold/40 px-2.5 py-1 font-display text-sm text-gold">
                  {n}
                </span>
              ))}
            </div>
          </div>
        )}

        {proposal.label_copy && (
          <blockquote className="border-l-2 border-gold pl-3 font-display text-base italic text-porcelain">
            “{proposal.label_copy}”
          </blockquote>
        )}

        {proposal.digital_mark && (
          <p className="text-xs text-supply">◈ {zh ? "数字印记" : "Digital Mark"}: {proposal.digital_mark}</p>
        )}
      </div>

      {cardUrl && (
        <div className="border-t border-hairline px-5 py-4">
          {/* Generated share card image */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={cardUrl} alt="Emotional card" className="w-full max-w-md rounded-lg border border-hairline" />
          <a href={cardUrl} download="zotaix-card.png" className="mt-2 inline-block text-xs text-gold hover:underline">
            {zh ? "下载卡片" : "Download card"}
          </a>
        </div>
      )}

      {!compact && (
        <div className="flex flex-wrap gap-2 border-t border-hairline bg-veil/30 px-5 py-4">
          {proposal.next_actions.map((key) => (
            <button
              key={key}
              onClick={() => handleAction(key)}
              disabled={saving === key}
              className="rounded-md border border-hairline px-3 py-1.5 text-xs text-mist transition-colors hover:border-gold hover:text-gold disabled:opacity-50"
            >
              {saving === key ? (zh ? "处理中…" : "Working…") : saved === key ? (zh ? "✓ 已完成" : "✓ Done") : zh ? ACTION_LABELS[key].zh : ACTION_LABELS[key].en}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
