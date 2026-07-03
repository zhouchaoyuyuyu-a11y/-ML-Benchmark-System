"use client";

import { useState } from "react";
import type { ChangeEvent, FormEvent, ReactNode } from "react";
import { Button, ButtonLink, Notice, StatusPill } from "@/components/ui";

export interface TradeDraftOption {
  id: string;
  title: string;
  object_type: string;
  budget?: string;
}

interface TradeApiResponse {
  ok?: boolean;
  error?: string;
  note?: string;
  request?: { id: string; human_review_status: string };
}

interface QuoteSuccess {
  id: string;
  status: string;
  note: string;
}

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div>
      <span className="text-xs uppercase tracking-wider text-mist">{label}</span>
      <div className="mt-1.5">{children}</div>
      {hint && <p className="mt-1 text-xs text-mist">{hint}</p>}
    </div>
  );
}

export default function TradeClient({
  zh = false,
  signedIn,
  drafts,
}: {
  zh?: boolean;
  signedIn: boolean;
  drafts: TradeDraftOption[];
}) {
  const [draftId, setDraftId] = useState(drafts[0]?.id ?? "");
  const [quantity, setQuantity] = useState("12");
  const [budget, setBudget] = useState("");
  const [deadline, setDeadline] = useState("");
  const [region, setRegion] = useState("");
  const [invoice, setInvoice] = useState(false);
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [needsAuth, setNeedsAuth] = useState(false);
  const [success, setSuccess] = useState<QuoteSuccess | null>(null);

  const onText =
    (setter: (v: string) => void) => (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
      setter(e.target.value);

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!draftId) {
      setError(zh ? "请选择一份草稿——报价始终附着在已保存的对象上。" : "Please select a draft — quotes always attach to a saved object.");
      return;
    }
    if (!budget.trim()) {
      setError(zh ? "请填写预算，礼宾以此为报价边界。" : "Please state a budget — the concierge quotes within it.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/trade", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          request_type: "quote",
          draftId,
          quantity: Math.max(1, Number(quantity) || 1),
          budget: budget.trim(),
          deadline: deadline || undefined,
          delivery_region: region.trim() || undefined,
          invoice_required: invoice,
          notes: notes.trim() || undefined,
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
      setError(zh ? "网络异常，请稍后再试。" : "A network hiccup interrupted the submission — please try again.");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setSuccess(null);
    setError(null);
    setQuantity("12");
    setBudget("");
    setDeadline("");
    setRegion("");
    setInvoice(false);
    setNotes("");
  }

  if (!signedIn || needsAuth) {
    return (
      <Notice tone="gold" title={zh ? "登录后申请报价" : "Sign in to request a quote"}>
        <p>
          {zh
            ? "报价请求会附着在你账户下的草稿上，并进入人工审核与合规审查。请先登录，或注册一个账户。"
            : "Quote requests attach to drafts saved under your account and enter human review and compliance checks. Sign in first, or create an account."}
        </p>
        <div className="mt-3 flex flex-wrap gap-3">
          <ButtonLink href="/login" variant="gold">
            {zh ? "去登录" : "Sign in"}
          </ButtonLink>
          <ButtonLink href="/forge" variant="outline">
            {zh ? "先去 Forge 创造" : "Create in Forge first"}
          </ButtonLink>
        </div>
      </Notice>
    );
  }

  if (drafts.length === 0) {
    return (
      <Notice tone="gold" title={zh ? "还没有可报价的草稿" : "No drafts to quote yet"}>
        <p>
          {zh
            ? "报价始终始于一个已保存的对象。先在 Forge 里把一种情绪变成酒体方向、命名与文案，保存后再回到这里申请报价。"
            : "A quote always starts from a saved object. Turn an emotion into a liquid direction, names, and label copy in Forge, save it, then return here to request a quote."}
        </p>
        <div className="mt-3">
          <ButtonLink href="/forge" variant="gold">
            {zh ? "去 Forge 创造一个对象" : "Create an object in Forge"}
          </ButtonLink>
        </div>
      </Notice>
    );
  }

  if (success) {
    const nextSteps = [
      zh ? "人工礼宾审核数量、预算、期限与地区" : "A human concierge reviews quantity, budget, deadline, and region",
      zh ? "合规团队核验标签与酒类规则" : "The compliance team verifies label and alcohol rules",
      zh ? "正式报价单送达后，由你决定是否推进" : "A formal quotation arrives — you decide whether to proceed",
      zh ? "交付后每件对象绑定档案记录" : "After delivery, every unit binds to a Reserve record",
    ];
    return (
      <div className="zx-card zx-fade-up p-6">
        <div className="flex flex-wrap items-center gap-3">
          <StatusPill status={success.status} />
          <p className="font-display text-lg text-porcelain">{zh ? "报价请求已受理" : "Quote request received"}</p>
        </div>
        {success.note && <p className="mt-3 text-sm leading-relaxed text-mist">{success.note}</p>}
        <p className="mt-2 text-xs text-mist">
          {zh ? "受理编号：" : "Reference: "}
          <span className="text-gold">{success.id}</span>
        </p>
        <ol className="mt-4 space-y-2">
          {nextSteps.map((step, i) => (
            <li key={step} className="flex items-start gap-3 text-sm text-mist">
              <span className="font-display text-gold/70">{String(i + 1).padStart(2, "0")}</span>
              <span>{step}</span>
            </li>
          ))}
        </ol>
        <div className="mt-5">
          <Button variant="outline" onClick={reset}>
            {zh ? "再提交一份请求" : "Submit another request"}
          </Button>
        </div>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="zx-card p-5 sm:p-6">
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="sm:col-span-2">
          <Field
            label={zh ? "选择草稿" : "Select a draft"}
            hint={zh ? "报价附着在这个对象上——其液体、香氛与视觉方向会随请求提交。" : "The quote attaches to this object — its liquid, scent, and visual directions travel with the request."}
          >
            <select value={draftId} onChange={onText(setDraftId)} className={inputCls}>
              {drafts.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.title} · {d.object_type}
                  {d.budget ? ` · ${d.budget}` : ""}
                </option>
              ))}
            </select>
          </Field>
        </div>
        <Field label={zh ? "数量" : "Quantity"}>
          <input
            type="number"
            min={1}
            value={quantity}
            onChange={onText(setQuantity)}
            className={inputCls}
            placeholder="12"
            required
          />
        </Field>
        <Field label={zh ? "预算" : "Budget"}>
          <input
            value={budget}
            onChange={onText(setBudget)}
            className={inputCls}
            maxLength={120}
            placeholder={zh ? "例如：7,200 元" : "e.g. 7,200 RMB"}
            required
          />
        </Field>
        <Field label={zh ? "交付期限" : "Deadline"}>
          <input type="date" value={deadline} onChange={onText(setDeadline)} className={inputCls} />
        </Field>
        <Field label={zh ? "交付地区" : "Delivery region"}>
          <input
            value={region}
            onChange={onText(setRegion)}
            className={inputCls}
            maxLength={200}
            placeholder={zh ? "例如：杭州" : "e.g. Hangzhou"}
          />
        </Field>
        <div className="sm:col-span-2">
          <label className="flex cursor-pointer items-center gap-3 rounded-lg border border-hairline px-4 py-3">
            <input
              type="checkbox"
              checked={invoice}
              onChange={(e) => setInvoice(e.target.checked)}
              className="h-4 w-4 accent-gold"
            />
            <span className="text-sm text-porcelain">{zh ? "需要开具发票" : "Invoice required"}</span>
          </label>
        </div>
        <div className="sm:col-span-2">
          <Field label={zh ? "备注" : "Notes"}>
            <textarea
              value={notes}
              onChange={onText(setNotes)}
              className={`${inputCls} min-h-24`}
              maxLength={800}
              placeholder={
                zh
                  ? "任何礼宾需要知道的事：赠送场合、刻字、分批交付偏好……"
                  : "Anything the concierge should know: the occasion, engraving, staged-delivery preferences…"
              }
            />
          </Field>
        </div>
      </div>

      {error && (
        <div className="mt-4">
          <Notice tone="ember">{error}</Notice>
        </div>
      )}

      <div className="mt-5 flex flex-wrap items-center gap-4">
        <Button type="submit" variant="gold" disabled={loading}>
          {loading ? (zh ? "提交中…" : "Submitting…") : zh ? "提交报价请求" : "Submit quote request"}
        </Button>
        <p className="text-xs leading-relaxed text-mist">
          {zh
            ? "提交后进入人工审核与合规审查，一个工作日内回复。"
            : "Enters human review and compliance checks; reply within one business day."}
        </p>
      </div>
    </form>
  );
}
