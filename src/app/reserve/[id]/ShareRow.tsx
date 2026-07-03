"use client";

import Link from "next/link";
import { useState } from "react";
import { Button, Notice, StatusPill } from "@/components/ui";

interface TradeApiResponse {
  ok?: boolean;
  error?: string;
  note?: string;
  request?: { id: string; human_review_status: string };
}

const linkBtnCls =
  "inline-flex items-center justify-center gap-2 rounded-md border border-hairline px-5 py-2.5 text-sm font-medium text-porcelain transition-colors hover:border-gold hover:text-gold";

/** Share, card download, and replenishment actions for a Reserve certificate. */
export default function ShareRow({
  zh = false,
  zotaixId,
  cardUrl,
  repurchaseEligible,
  isPublic,
}: {
  zh?: boolean;
  zotaixId: string;
  cardUrl: string;
  repurchaseEligible: boolean;
  isPublic: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const [replenishing, setReplenishing] = useState(false);
  const [needsAuth, setNeedsAuth] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<{ id: string; status: string; note: string } | null>(null);

  async function copyLink() {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2400);
    } catch {
      setError(zh ? "复制没有成功——请从地址栏手动复制链接。" : "Copying did not go through — please copy the link from the address bar.");
    }
  }

  async function requestReplenishment() {
    setReplenishing(true);
    setError(null);
    try {
      const res = await fetch("/api/trade", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ request_type: "replenishment", notes: zotaixId }),
      });
      if (res.status === 401) {
        setNeedsAuth(true);
        return;
      }
      const data = (await res.json().catch(() => ({}))) as TradeApiResponse;
      if (!res.ok || !data.ok || !data.request) {
        setError(data.error ?? (zh ? "提交未成功，请稍后再试。" : "The request could not be submitted — please try again."));
        return;
      }
      setSuccess({
        id: data.request.id,
        status: data.request.human_review_status || "pending",
        note: data.note ?? "",
      });
    } catch {
      setError(zh ? "网络异常，请稍后再试。" : "A network hiccup interrupted the request — please try again.");
    } finally {
      setReplenishing(false);
    }
  }

  return (
    <div className="zx-card p-5 sm:p-6">
      <p className="font-display text-base text-porcelain">{zh ? "分享与补铸" : "Share & replenish"}</p>
      <p className="mt-1.5 text-xs leading-relaxed text-mist">
        {isPublic
          ? zh
            ? "公开证书可被任何人访问——把这个时刻交给该读到它的人。"
            : "Public certificates open for anyone — hand this moment to the person meant to read it."
          : zh
            ? "私密档案的链接仅你本人可打开；情绪卡片仍可下载与分享。"
            : "A private record's link opens only for you; the emotional card can still be downloaded and shared."}
      </p>

      <div className="mt-4 flex flex-wrap gap-2.5">
        <Button variant="outline" onClick={copyLink}>
          {copied ? (zh ? "✓ 已复制链接" : "✓ Link copied") : zh ? "复制证书链接" : "Copy certificate link"}
        </Button>
        <a href={cardUrl} download={`${zotaixId}.png`} className={linkBtnCls}>
          {zh ? "下载情绪卡片" : "Download emotional card"}
        </a>
        {repurchaseEligible && !success && (
          <Button variant="gold" onClick={requestReplenishment} disabled={replenishing}>
            {replenishing ? (zh ? "提交中…" : "Submitting…") : zh ? "发起补铸" : "Request replenishment"}
          </Button>
        )}
      </div>

      <p className="mt-3 text-xs leading-relaxed text-mist">
        {zh ? "想在微信里分享？" : "Sharing inside WeChat?"}{" "}
        <Link href="/wechat" className="text-gold hover:underline">
          {zh ? "关注 ZOTAIX 卓序公众号 →" : "Follow the ZOTAIX official account →"}
        </Link>{" "}
        {zh
          ? "在公众号会话中粘贴链接，即可生成带证书封面的分享卡。"
          : "Paste the link in the official-account chat to share it with a certificate cover."}
      </p>

      {success && (
        <div className="zx-fade-up mt-4 rounded-lg border border-hairline bg-obsidian/60 p-4">
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill status={success.status} />
            <p className="text-sm font-medium text-porcelain">
              {zh ? "补铸请求已受理" : "Replenishment request received"}
            </p>
          </div>
          <p className="mt-2 text-xs text-mist">
            {zh ? "受理编号：" : "Reference: "}
            <span className="text-gold">{success.id}</span> · {zotaixId}
          </p>
          {success.note && <p className="mt-2 text-xs leading-relaxed text-mist">{success.note}</p>}
        </div>
      )}

      {needsAuth && (
        <div className="mt-4">
          <Notice tone="gold" title={zh ? "登录后发起补铸" : "Sign in to request replenishment"}>
            {zh
              ? "补铸请求会记录在你的账户下，由人工礼宾审核数量、地区与合规后报价。"
              : "Replenishment requests are recorded under your account; a human concierge reviews quantity, region, and compliance before quoting."}{" "}
            <Link href="/login" className="text-gold hover:underline">
              {zh ? "去登录 →" : "Sign in →"}
            </Link>
          </Notice>
        </div>
      )}

      {error && (
        <div className="mt-4">
          <Notice tone="ember">{error}</Notice>
        </div>
      )}
    </div>
  );
}
