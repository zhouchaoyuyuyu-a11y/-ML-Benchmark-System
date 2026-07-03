import { Stat, Tag } from "@/components/ui";
import { dataMode, db } from "@/lib/store";
import SettingsClient from "./SettingsClient";

export default async function AdminSettingsPage() {
  const data = db();
  const settings = data.settings;
  const mode = dataMode();

  return (
    <div className="max-w-6xl">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="font-display text-2xl text-porcelain">Global Settings</h1>
        {mode === "memory" ? (
          <Tag tone="supply">seeded in-memory store</Tag>
        ) : (
          <Tag tone="jade">database connected</Tag>
        )}
      </div>
      <p className="mt-1 text-sm text-mist">
        Platform-wide levers — daily chat quotas, Core Sequence pricing, co-creation thresholds, the age gate,
        and the brand line. Quota checks, the membership page, and the co-creation pool all read these values
        on every request.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Site" value={settings.site_name} hint="Platform identity" />
        <Stat
          label="Guest daily chat"
          value={String(settings.guest_daily_chat)}
          hint="Concierge turns before signing up"
        />
        <Stat
          label="Pro price · quarter"
          value={`¥${settings.pro_price_quarter}`}
          hint="Core Sequence Pro quarterly"
        />
        <Stat
          label="Age gate"
          value={settings.age_gate_enabled ? "On" : "Off"}
          hint="Alcohol-page confirmation overlay"
        />
      </div>

      <div className="mt-8 pb-6">
        <SettingsClient settings={settings} mode={mode} />
      </div>
    </div>
  );
}
