import { Stat, Tag } from "@/components/ui";
import { integrationStatus } from "@/lib/config";
import { db } from "@/lib/store";
import WechatConfigClient from "./WechatConfigClient";

export default async function AdminWechatPage() {
  const data = db();
  const config = data.wechat_config;
  const integrations = integrationStatus().filter((i) => i.key === "wechat" || i.key === "wechat_pay");
  const wechatEnv = integrations.find((i) => i.key === "wechat");
  const menuEntries = config.menu_config.reduce((sum, g) => sum + g.children.length, 0);

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">WeChat Official Account</h1>
      <p className="mt-1 text-sm text-mist">
        Configuration for the ZOTAIX 卓序 official account — menu structure, auto-replies, QR entry, and
        customer-service routing. This record is the single source of truth: the public /wechat page renders
        from it, and the operations team pushes it to the WeChat API once credentials are live.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Account" value={config.official_account_name} hint={config.enabled ? "Channel enabled" : "Channel disabled"} />
        <Stat label="Menu groups" value={String(config.menu_config.length)} hint="Top-level menus (max 3)" />
        <Stat label="Menu entries" value={String(menuEntries)} hint="Sub-items across groups" />
        <Stat label="Auto-replies" value={String(config.auto_reply_config.length)} hint="Keyword & follow triggers" />
      </div>

      <section className="mt-8">
        <h2 className="font-display text-lg text-porcelain">Integration status</h2>
        <p className="mt-1 text-xs text-mist">
          WeChat credentials are read from the environment only and never stored in this database.
        </p>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          {integrations.map((i) => (
            <div key={i.key} className="zx-card p-4">
              <div className="flex items-start justify-between gap-3">
                <p className="text-sm text-porcelain">{i.label}</p>
                {i.configured ? <Tag tone="jade">configured</Tag> : <Tag tone="gold">configuration needed</Tag>}
              </div>
              <p className="mt-2 text-xs leading-relaxed text-mist">
                {i.configured
                  ? "Live credentials detected — publishing and API calls use the external service."
                  : `Active fallback: ${i.fallback}`}
              </p>
              <p className="mt-2 font-mono text-[10px] tracking-wide text-mist/70">{i.envVars.join(" · ")}</p>
            </div>
          ))}
        </div>
      </section>

      <div className="mt-8 pb-6">
        <WechatConfigClient config={config} envConfigured={wechatEnv?.configured ?? false} />
      </div>
    </div>
  );
}
