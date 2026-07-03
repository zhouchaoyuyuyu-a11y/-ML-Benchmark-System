"use client";

import { useState } from "react";
import { Button, ButtonLink, Notice, StatusPill } from "@/components/ui";

interface TradeApiResponse {
  ok?: boolean;
  error?: string;
  note?: string;
  request?: { id: string; human_review_status: string };
}

interface AuthSuccess {
  id: string;
  status: string;
  note: string;
}

export default function MarketActions({
  zh = false,
  itemId,
  itemTitle,
  draftId,
  projectId,
}: {
  zh?: boolean;
  itemId: string;
  itemTitle: string;
  draftId?: string;
  projectId?: string;
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [needsAuth, setNeedsAuth] = useState(false);
  const [success, setSuccess] = useState<AuthSuccess | null>(null);

  async function requestAuthorization() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/trade", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          request_type: "authorization",
          draftId: draftId || undefined,
          notes: `Market item ${itemId} — authorization request for “${itemTitle}”`,
        }),
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
      setLoading(false);
    }
  }

  return (
    <div className="zx-card p-5 sm:p-6">
      <p className="font-display text-base text-porcelain">{zh ? "对这条陈列采取行动" : "Act on this listing"}</p>
      <p className="mt-2 text-xs leading-relaxed text-mist">
        {zh
          ? "授权申请进入人工审核；共创通过项目页加入；企业需求交给 Maison 礼宾。"
          : "Authorization requests enter human review; co-creation joins happen on the project page; enterprise needs go to the Maison concierge."}
      </p>

      {success ? (
        <div className="zx-fade-up mt-4 rounded-lg border border-hairline bg-obsidian/60 p-4">
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill status={success.status} />
            <p className="text-sm font-medium text-porcelain">{zh ? "授权申请已受理" : "Authorization request received"}</p>
          </div>
          {success.note && <p className="mt-2 text-xs leading-relaxed text-mist">{success.note}</p>}
          <p className="mt-2 text-xs text-mist">
            {zh ? "受理编号：" : "Reference: "}
            <span className="text-gold">{success.id}</span>
          </p>
          <p className="mt-2 text-xs leading-relaxed text-mist">
            {zh
              ? "礼宾团队会审核使用范围与分成协议，并在一个工作日内联系你。"
              : "The concierge team reviews scope and the income-share agreement, and contacts you within one business day."}
          </p>
        </div>
      ) : (
        <div className="mt-4 flex flex-col gap-2.5">
          <Button variant="gold" onClick={requestAuthorization} disabled={loading} className="w-full">
            {loading ? (zh ? "提交中…" : "Submitting…") : zh ? "申请授权" : "Request authorization"}
          </Button>
          {projectId && (
            <ButtonLink href={`/co-create/${projectId}`} variant="outline" className="w-full">
              {zh ? "加入共创" : "Join co-creation"}
            </ButtonLink>
          )}
          <ButtonLink href="/maison#enterprise" variant="outline" className="w-full">
            {zh ? "企业询价" : "Enterprise inquiry"}
          </ButtonLink>
        </div>
      )}

      {needsAuth && (
        <div className="mt-4">
          <Notice tone="gold" title={zh ? "登录后申请授权" : "Sign in to request authorization"}>
            {zh
              ? "授权申请会记录在你的账户下并进入人工审核。"
              : "Authorization requests are recorded under your account and enter human review."}{" "}
            <ButtonLink href="/login" variant="ghost" className="!px-0 text-gold">
              {zh ? "去登录 →" : "Sign in →"}
            </ButtonLink>
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
