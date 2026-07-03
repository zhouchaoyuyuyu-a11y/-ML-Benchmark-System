"use client";

import { useState } from "react";
import type { FormEvent } from "react";
import { Button, Notice } from "@/components/ui";

/* Overseas email subscription. Records the lead server-side via /api/trade so
   the concierge team sees every subscriber in the same inbox as collaboration
   inquiries. */

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

export default function SubscribeForm({ zh = false }: { zh?: boolean }) {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const trimmed = email.trim();
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(trimmed)) {
      setError(zh ? "请输入有效的邮箱地址。" : "Please enter a valid email address.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/trade", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          request_type: "collaboration",
          contact: trimmed,
          organization: "newsletter",
          notes: "email subscription",
          name: "Newsletter subscriber",
        }),
      });
      const data = (await res.json().catch(() => ({}))) as { ok?: boolean; error?: string };
      if (!res.ok || !data.ok) {
        setError(
          data.error ?? (zh ? "订阅未成功，请稍后再试。" : "The subscription could not be recorded — please try again.")
        );
        return;
      }
      setDone(true);
    } catch {
      setError(zh ? "网络异常，请稍后再试。" : "A network hiccup interrupted the request — please try again.");
    } finally {
      setLoading(false);
    }
  }

  if (done) {
    return (
      <div className="zx-card zx-fade-up flex items-start gap-3 p-5">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-jade/40 bg-jade/10 text-jade">
          ✓
        </span>
        <div>
          <p className="text-sm font-medium text-porcelain">{zh ? "订阅成功 ✓" : "Subscribed ✓"}</p>
          <p className="mt-1 text-xs leading-relaxed text-mist">
            {zh
              ? "你的邮箱已记录在案。品牌志、共创进展与档案故事会随发布节奏送达；每封邮件底部都有一键退订。"
              : "Your address is on record. Journal issues, co-creation progress, and Reserve stories arrive on the publishing rhythm — every email carries a one-click unsubscribe."}
          </p>
        </div>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="zx-card p-5">
      <label htmlFor="newsletter-email" className="text-xs uppercase tracking-wider text-mist">
        {zh ? "邮箱地址" : "Email address"}
      </label>
      <div className="mt-2 flex flex-col gap-3 sm:flex-row">
        <input
          id="newsletter-email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className={inputCls}
          placeholder="you@example.com"
          maxLength={200}
          required
        />
        <Button type="submit" variant="gold" disabled={loading} className="shrink-0">
          {loading ? (zh ? "订阅中…" : "Subscribing…") : zh ? "订阅" : "Subscribe"}
        </Button>
      </div>
      {error && (
        <div className="mt-3">
          <Notice tone="ember">{error}</Notice>
        </div>
      )}
      <p className="mt-3 text-xs leading-relaxed text-mist">
        {zh
          ? "仅用于品牌通讯，不会用于其他用途；随时可退订。"
          : "Used for the brand letter only, nothing else — unsubscribe anytime."}
      </p>
    </form>
  );
}
