"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button, Notice } from "@/components/ui";
import type { PlatformSettings } from "@/lib/types";

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

type NumKey =
  | "guest_daily_chat"
  | "free_daily_chat"
  | "lite_daily_chat"
  | "pro_daily_chat"
  | "lite_monthly_proposals"
  | "pro_monthly_proposals"
  | "lite_price_month"
  | "lite_price_quarter"
  | "pro_price_month"
  | "pro_price_quarter"
  | "co_create_public_threshold"
  | "co_create_review_threshold"
  | "co_create_label_threshold"
  | "co_create_flavor_threshold"
  | "co_create_enterprise_threshold"
  | "co_create_supply_threshold"
  | "co_create_partner_threshold";

interface NumField {
  key: NumKey;
  label: string;
  hint: string;
}

const quotaFields: NumField[] = [
  { key: "guest_daily_chat", label: "Guest daily chat", hint: "Concierge turns per day for visitors" },
  { key: "free_daily_chat", label: "Free daily chat", hint: "Registered accounts without a plan" },
  { key: "lite_daily_chat", label: "Lite daily chat", hint: "Core Sequence Lite members" },
  { key: "pro_daily_chat", label: "Pro daily chat", hint: "Core Sequence Pro members" },
  { key: "lite_monthly_proposals", label: "Lite monthly proposals", hint: "Structured generations per month" },
  { key: "pro_monthly_proposals", label: "Pro monthly proposals", hint: "Structured generations per month" },
];

const priceFields: NumField[] = [
  { key: "lite_price_month", label: "Lite · monthly (CNY)", hint: "Billed each month" },
  { key: "lite_price_quarter", label: "Lite · quarterly (CNY)", hint: "Billed each quarter" },
  { key: "pro_price_month", label: "Pro · monthly (CNY)", hint: "Billed each month" },
  { key: "pro_price_quarter", label: "Pro · quarterly (CNY)", hint: "Billed each quarter" },
];

const thresholdFields: NumField[] = [
  { key: "co_create_public_threshold", label: "Public page", hint: "Supporters to open a public project page" },
  { key: "co_create_review_threshold", label: "Concept review", hint: "Units to enter human concept review" },
  { key: "co_create_label_threshold", label: "Label theming", hint: "Units to unlock label & gift-box theming" },
  { key: "co_create_flavor_threshold", label: "Flavor review", hint: "Units to unlock flavor-direction review" },
  { key: "co_create_enterprise_threshold", label: "Enterprise review", hint: "Units to enter enterprise gifting review" },
  { key: "co_create_supply_threshold", label: "Supply line run", hint: "Units to qualify for a Supply production run" },
  { key: "co_create_partner_threshold", label: "Partner program", hint: "Units to open partner and brand terms" },
];

const allNumFields = [...quotaFields, ...priceFields, ...thresholdFields];

export default function SettingsClient({
  settings,
  mode,
}: {
  settings: PlatformSettings;
  mode: "memory" | "database";
}) {
  const router = useRouter();
  const [values, setValues] = useState<Record<NumKey, string>>(
    Object.fromEntries(allNumFields.map((f) => [f.key, String(settings[f.key])])) as Record<NumKey, string>,
  );
  const [brandEn, setBrandEn] = useState(settings.brand_line_en);
  const [brandZh, setBrandZh] = useState(settings.brand_line_zh);
  const [ageGate, setAgeGate] = useState(settings.age_gate_enabled);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function setNum(key: NumKey, value: string) {
    setValues((v) => ({ ...v, [key]: value }));
    setSaved(false);
    setError(null);
  }

  async function save() {
    setSaved(false);
    setError(null);
    const numericPatch: Partial<Record<NumKey, number>> = {};
    for (const field of allNumFields) {
      const n = Number(values[field.key]);
      if (!Number.isFinite(n) || n < 0) {
        setError(`“${field.label}” must be a non-negative number.`);
        return;
      }
      numericPatch[field.key] = n;
    }
    setSaving(true);
    try {
      const res = await fetch("/api/admin/config", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          section: "settings",
          patch: {
            ...numericPatch,
            brand_line_en: brandEn,
            brand_line_zh: brandZh,
            age_gate_enabled: ageGate,
          },
        }),
      });
      const data = await res.json().catch(() => ({ ok: false }));
      if (!data.ok) {
        setError(data.error ?? "Save failed — check the values and try again.");
      } else {
        setSaved(true);
        router.refresh();
      }
    } catch {
      setError("Network error — the settings were not saved.");
    } finally {
      setSaving(false);
    }
  }

  function numGrid(fields: NumField[], cols: string) {
    return (
      <div className={`mt-4 grid gap-4 ${cols}`}>
        {fields.map((f) => (
          <label key={f.key} className="block">
            <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">{f.label}</span>
            <input
              type="number"
              min={0}
              value={values[f.key]}
              onChange={(e) => setNum(f.key, e.target.value)}
              className={inputCls}
            />
            <span className="mt-1 block text-[11px] text-mist/80">{f.hint}</span>
          </label>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="zx-card p-5">
          <h2 className="font-display text-lg text-porcelain">Quota</h2>
          <p className="mt-1 text-xs text-mist">
            Daily concierge turns and monthly structured proposals per tier — enforced by every /api/ai/generate call.
          </p>
          {numGrid(quotaFields, "sm:grid-cols-2")}
        </div>

        <div className="zx-card p-5">
          <h2 className="font-display text-lg text-porcelain">Pricing</h2>
          <p className="mt-1 text-xs text-mist">
            Core Sequence subscription prices in CNY — the membership page and subscribe flow read these values.
          </p>
          {numGrid(priceFields, "sm:grid-cols-2")}
        </div>
      </div>

      <div className="zx-card p-5">
        <h2 className="font-display text-lg text-porcelain">Co-creation thresholds</h2>
        <p className="mt-1 text-xs text-mist">
          Unit counts at which a co-creation project unlocks each stage — depth of customization scales with
          commitment, and every stage still passes human review.
        </p>
        {numGrid(thresholdFields, "sm:grid-cols-2 lg:grid-cols-4")}
      </div>

      <div className="zx-card p-5">
        <h2 className="font-display text-lg text-porcelain">Brand & compliance</h2>
        <p className="mt-1 text-xs text-mist">
          The one-paragraph brand line used across SEO descriptions and about surfaces, plus the platform-wide
          age gate.
        </p>
        <div className="mt-4 space-y-4">
          <label className="block">
            <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Brand line · English</span>
            <textarea
              value={brandEn}
              onChange={(e) => {
                setBrandEn(e.target.value);
                setSaved(false);
                setError(null);
              }}
              rows={3}
              className={`${inputCls} resize-y`}
            />
          </label>
          <label className="block">
            <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Brand line · 中文</span>
            <textarea
              value={brandZh}
              onChange={(e) => {
                setBrandZh(e.target.value);
                setSaved(false);
                setError(null);
              }}
              rows={3}
              className={`${inputCls} resize-y`}
            />
          </label>
          <label className="flex cursor-pointer items-start gap-3 rounded-md border border-hairline px-3 py-3 transition-colors hover:border-gold/40">
            <input
              type="checkbox"
              checked={ageGate}
              onChange={(e) => {
                setAgeGate(e.target.checked);
                setSaved(false);
                setError(null);
              }}
              className="mt-0.5 h-4 w-4 rounded border-hairline bg-ink accent-gold"
            />
            <span>
              <span className="block text-sm text-porcelain">Age gate enabled</span>
              <span className="mt-0.5 block text-xs text-mist">
                Shows the confirm-your-age overlay on alcohol-related pages (Maison, Forge, Trade, market detail).
              </span>
            </span>
          </label>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={save} disabled={saving}>
          {saving ? "Saving…" : "Save settings"}
        </Button>
        {saved && <span className="text-sm font-medium text-jade">Saved ✓</span>}
        {error && <span className="text-sm text-ember">{error}</span>}
      </div>

      <Notice tone={mode === "memory" ? "supply" : "gold"} title="Persistence">
        {mode === "memory"
          ? "This deployment runs on the seeded in-memory store: saves apply immediately to every page but reset when the server instance restarts. Set DATABASE_URL to persist settings in Postgres."
          : "This deployment is connected to a database: saves persist across restarts and apply to every page on the next request."}
      </Notice>
    </div>
  );
}
