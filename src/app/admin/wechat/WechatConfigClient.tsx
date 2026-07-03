"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button, Notice, Tag } from "@/components/ui";
import type { WechatConfig, WechatMenuItem } from "@/lib/types";

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";
const smallInputCls =
  "w-full rounded-md border border-hairline bg-ink px-2.5 py-2 text-xs text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

export default function WechatConfigClient({
  config,
  envConfigured,
}: {
  config: WechatConfig;
  envConfigured: boolean;
}) {
  const router = useRouter();
  const [name, setName] = useState(config.official_account_name);
  const [qrUrl, setQrUrl] = useState(config.qr_code_url ?? "");
  const [serviceUrl, setServiceUrl] = useState(config.customer_service_url ?? "");
  const [enabled, setEnabled] = useState(config.enabled);
  const [menu, setMenu] = useState<WechatMenuItem[]>(
    config.menu_config.map((g) => ({ label: g.label, children: g.children.map((c) => ({ ...c })) })),
  );
  const [replies, setReplies] = useState(config.auto_reply_config.map((r) => ({ ...r })));
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function touch() {
    setSaved(false);
    setError(null);
  }

  function setGroupLabel(gi: number, label: string) {
    touch();
    setMenu((m) => m.map((g, i) => (i === gi ? { ...g, label } : g)));
  }
  function setItemField(gi: number, ci: number, field: "label" | "target", value: string) {
    touch();
    setMenu((m) =>
      m.map((g, i) =>
        i === gi
          ? { ...g, children: g.children.map((c, j) => (j === ci ? { ...c, [field]: value } : c)) }
          : g,
      ),
    );
  }
  function addItem(gi: number) {
    touch();
    setMenu((m) =>
      m.map((g, i) =>
        i === gi && g.children.length < 4
          ? { ...g, children: [...g.children, { label: "", target: "/" }] }
          : g,
      ),
    );
  }
  function removeItem(gi: number, ci: number) {
    touch();
    setMenu((m) => m.map((g, i) => (i === gi ? { ...g, children: g.children.filter((_, j) => j !== ci) } : g)));
  }
  function addGroup() {
    touch();
    setMenu((m) => (m.length < 3 ? [...m, { label: "New group", children: [{ label: "", target: "/" }] }] : m));
  }

  function setReplyField(ri: number, field: "trigger" | "reply", value: string) {
    touch();
    setReplies((rs) => rs.map((r, i) => (i === ri ? { ...r, [field]: value } : r)));
  }
  function addReply() {
    touch();
    setReplies((rs) => [...rs, { trigger: "", reply: "" }]);
  }
  function removeReply(ri: number) {
    touch();
    setReplies((rs) => rs.filter((_, i) => i !== ri));
  }

  async function save() {
    setSaving(true);
    setSaved(false);
    setError(null);
    try {
      const res = await fetch("/api/admin/config", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          section: "wechat",
          patch: {
            official_account_name: name,
            qr_code_url: qrUrl,
            customer_service_url: serviceUrl,
            enabled,
            menu_config: menu,
            auto_reply_config: replies,
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
      setError("Network error — the configuration was not saved.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 lg:grid-cols-2">
        {/* Account basics */}
        <div className="zx-card p-5">
          <h2 className="font-display text-lg text-porcelain">Account</h2>
          <p className="mt-1 text-xs text-mist">Display name, QR entry image, and customer-service routing.</p>
          <div className="mt-4 space-y-4">
            <label className="block">
              <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Official account name</span>
              <input
                value={name}
                onChange={(e) => {
                  touch();
                  setName(e.target.value);
                }}
                className={inputCls}
                placeholder="ZOTAIX 卓序"
              />
            </label>
            <label className="block">
              <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Follow QR code URL</span>
              <input
                value={qrUrl}
                onChange={(e) => {
                  touch();
                  setQrUrl(e.target.value);
                }}
                className={inputCls}
                placeholder="https://… (leave empty to render the generated ZOTAIX identity mark)"
              />
            </label>
            <label className="block">
              <span className="mb-1.5 block text-xs uppercase tracking-wider text-mist">Customer service URL</span>
              <input
                value={serviceUrl}
                onChange={(e) => {
                  touch();
                  setServiceUrl(e.target.value);
                }}
                className={inputCls}
                placeholder="https://… (leave empty to route 客服 replies to concierge email)"
              />
            </label>
            <label className="flex cursor-pointer items-center gap-2.5 text-sm text-porcelain">
              <input
                type="checkbox"
                checked={enabled}
                onChange={(e) => {
                  touch();
                  setEnabled(e.target.checked);
                }}
                className="h-4 w-4 rounded border-hairline bg-ink accent-gold"
              />
              <span>Channel enabled</span>
              <span className="text-xs text-mist">— controls the public /wechat page and nav entry</span>
            </label>
          </div>
        </div>

        {/* Credentials */}
        <div className="zx-card p-5">
          <div className="flex items-start justify-between gap-3">
            <h2 className="font-display text-lg text-porcelain">API credentials</h2>
            {envConfigured ? <Tag tone="jade">configured</Tag> : <Tag tone="gold">awaiting credentials</Tag>}
          </div>
          <p className="mt-1 text-xs text-mist">
            AppID and AppSecret are read exclusively from the deployment environment. They are never stored in
            this database, never displayed here, and never editable through this console.
          </p>
          <dl className="mt-4 space-y-2">
            <div className="flex items-center justify-between gap-3 rounded-md border border-hairline px-3 py-2.5">
              <dt className="font-mono text-xs text-porcelain">WECHAT_APP_ID</dt>
              <dd className={`text-xs font-medium ${envConfigured ? "text-jade" : "text-gold"}`}>
                {envConfigured ? "set" : "not set"}
              </dd>
            </div>
            <div className="flex items-center justify-between gap-3 rounded-md border border-hairline px-3 py-2.5">
              <dt className="font-mono text-xs text-porcelain">WECHAT_APP_SECRET</dt>
              <dd className={`text-xs font-medium ${envConfigured ? "text-jade" : "text-gold"}`}>
                {envConfigured ? "set" : "not set"}
              </dd>
            </div>
          </dl>
          <p className="mt-3 text-xs leading-relaxed text-mist">
            {envConfigured
              ? "Live credentials detected — the menu and auto-replies below can be pushed to the WeChat Official Account API on the next publish run."
              : "Until both variables are present, the account runs in preview: the public page shows the QR entry and this exact menu and reply set, so followers see the same structure the API will receive."}
          </p>
        </div>
      </div>

      {/* Menu editor */}
      <div className="zx-card p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="font-display text-lg text-porcelain">Custom menu</h2>
            <p className="mt-1 text-xs text-mist">
              WeChat allows up to 3 top-level groups with up to 4 entries each. Targets are site paths or full URLs.
            </p>
          </div>
          {menu.length < 3 && (
            <Button variant="outline" className="!px-3 !py-1.5 !text-xs" onClick={addGroup}>
              + Add group
            </Button>
          )}
        </div>
        <div className="mt-4 grid gap-4 lg:grid-cols-3">
          {menu.map((group, gi) => (
            <div key={gi} className="rounded-lg border border-hairline p-4">
              <label className="block">
                <span className="mb-1.5 block text-[10px] uppercase tracking-wider text-mist">Group {gi + 1} label</span>
                <input
                  value={group.label}
                  onChange={(e) => setGroupLabel(gi, e.target.value)}
                  className={smallInputCls}
                  placeholder="Group label"
                />
              </label>
              <div className="mt-3 space-y-3">
                {group.children.map((item, ci) => (
                  <div key={ci} className="rounded-md border border-hairline/60 bg-ink/60 p-2.5">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[10px] uppercase tracking-wider text-mist">Entry {ci + 1}</span>
                      <button
                        type="button"
                        onClick={() => removeItem(gi, ci)}
                        className="rounded border border-ember/40 px-1.5 py-0.5 text-[10px] text-ember transition-colors hover:bg-ember/10"
                      >
                        Remove
                      </button>
                    </div>
                    <input
                      value={item.label}
                      onChange={(e) => setItemField(gi, ci, "label", e.target.value)}
                      className={`${smallInputCls} mt-2`}
                      placeholder="Label"
                    />
                    <input
                      value={item.target}
                      onChange={(e) => setItemField(gi, ci, "target", e.target.value)}
                      className={`${smallInputCls} mt-2 font-mono`}
                      placeholder="/concierge?mode=daily"
                    />
                  </div>
                ))}
              </div>
              {group.children.length < 4 && (
                <button
                  type="button"
                  onClick={() => addItem(gi)}
                  className="mt-3 w-full rounded-md border border-dashed border-hairline px-2 py-1.5 text-xs text-mist transition-colors hover:border-gold hover:text-gold"
                >
                  + Add entry ({group.children.length}/4)
                </button>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Auto replies */}
      <div className="zx-card p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="font-display text-lg text-porcelain">Auto-replies</h2>
            <p className="mt-1 text-xs text-mist">
              Keyword triggers and their replies. The reserved trigger <code className="font-mono text-gold">__follow__</code> is
              the welcome message sent when someone follows the account.
            </p>
          </div>
          <Button variant="outline" className="!px-3 !py-1.5 !text-xs" onClick={addReply}>
            + Add reply
          </Button>
        </div>
        <div className="mt-4 space-y-3">
          {replies.map((r, ri) => (
            <div key={ri} className="flex flex-col gap-2 rounded-md border border-hairline p-3 sm:flex-row sm:items-start">
              <input
                value={r.trigger}
                onChange={(e) => setReplyField(ri, "trigger", e.target.value)}
                className={`${smallInputCls} sm:w-40 sm:shrink-0`}
                placeholder="Trigger keyword"
              />
              <textarea
                value={r.reply}
                onChange={(e) => setReplyField(ri, "reply", e.target.value)}
                rows={2}
                className={`${smallInputCls} flex-1 resize-y`}
                placeholder="Reply text"
              />
              <button
                type="button"
                onClick={() => removeReply(ri)}
                className="self-start rounded border border-ember/40 px-2 py-1 text-[10px] text-ember transition-colors hover:bg-ember/10"
              >
                Remove
              </button>
            </div>
          ))}
          {replies.length === 0 && (
            <p className="rounded-md border border-dashed border-hairline px-3 py-4 text-center text-xs text-mist">
              No auto-replies yet — add a trigger above so followers always get an answer.
            </p>
          )}
        </div>
      </div>

      {/* Save bar */}
      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={save} disabled={saving}>
          {saving ? "Saving…" : "Save configuration"}
        </Button>
        {saved && <span className="text-sm font-medium text-jade">Saved ✓</span>}
        {error && <span className="text-sm text-ember">{error}</span>}
      </div>

      <Notice tone="gold" title="Publishing flow">
        Menu structure and auto-replies are stored server-side as the single source of truth. Once
        WECHAT_APP_ID and WECHAT_APP_SECRET are present in the environment, the operations team pushes this
        exact record to the WeChat Official Account API in one publish run — the public /wechat page always
        previews the same structure followers will see inside WeChat.
      </Notice>
    </div>
  );
}
