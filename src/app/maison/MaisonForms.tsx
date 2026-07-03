"use client";

import { useState } from "react";
import type { ChangeEvent, FormEvent, ReactNode } from "react";
import { Button, Notice, StatusPill } from "@/components/ui";

type FormKind = "enterprise" | "collaboration" | "concierge";

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

const ENTERPRISE_SCENARIOS: { value: string; en: string; zh: string }[] = [
  { value: "Enterprise gifting", en: "Enterprise gifting", zh: "企业礼赠" },
  { value: "Client appreciation", en: "Client appreciation", zh: "客户答谢" },
  { value: "Banquet", en: "Banquet", zh: "高端宴席" },
  { value: "Private celebration", en: "Private celebration", zh: "私人庆典" },
  { value: "Hotels & clubs", en: "Hotels & clubs", zh: "酒店与会所" },
  { value: "City souvenir", en: "City souvenir", zh: "城市伴手礼" },
  { value: "Cultural tourism", en: "Cultural tourism", zh: "文旅项目" },
  { value: "Wedding", en: "Wedding", zh: "婚礼" },
  { value: "Anniversary", en: "Anniversary", zh: "周年纪念" },
  { value: "Collection", en: "Collection", zh: "收藏" },
];

interface TradeSuccess {
  id: string;
  status: string;
  note: string;
}

interface TradeResponse {
  ok?: boolean;
  error?: string;
  note?: string;
  request?: { id: string; human_review_status: string };
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div>
      <span className="text-xs uppercase tracking-wider text-mist">{label}</span>
      <div className="mt-1.5">{children}</div>
      {hint && <p className="mt-1 text-xs text-mist">{hint}</p>}
    </div>
  );
}

export default function MaisonForms({ zh = false, form }: { zh?: boolean; form: FormKind }) {
  const [fields, setFields] = useState<Record<string, string>>({
    scenario: form === "enterprise" ? "Enterprise gifting" : "",
  });
  const [invoice, setInvoice] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<TradeSuccess | null>(null);

  const set =
    (key: string) => (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
      setFields((f) => ({ ...f, [key]: e.target.value }));

  const v = (key: string) => fields[key] ?? "";

  function buildPayload(): Record<string, unknown> {
    if (form === "enterprise") {
      return {
        request_type: "enterprise",
        name: v("name"),
        organization: v("organization"),
        contact: v("contact"),
        scenario: v("scenario"),
        quantity: Math.max(1, Number(v("quantity")) || 1),
        budget: v("budget") || undefined,
        deadline: v("deadline") || undefined,
        delivery_region: v("delivery_region") || undefined,
        invoice_required: invoice,
        liquid_direction: v("liquid_direction") || undefined,
        scent_direction: v("scent_direction") || undefined,
        bottle_direction: v("bottle_direction") || undefined,
        packaging_direction: v("packaging_direction") || undefined,
        sample_path: v("sample_path") || undefined,
        logistics_notes: v("logistics_notes") || undefined,
      };
    }
    if (form === "collaboration") {
      return {
        request_type: "collaboration",
        organization: v("organization"),
        contact: v("contact"),
        scenario: v("scenario"),
        budget: v("budget") || undefined,
      };
    }
    return {
      request_type: "enterprise",
      name: v("name"),
      contact: v("contact"),
      scenario: v("scenario"),
      budget: v("budget") || undefined,
      notes: "private concierge",
    };
  }

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!v("contact").trim()) {
      setError(
        zh
          ? "请留下邮箱或电话，礼宾才能在一个工作日内回复你。"
          : "Please leave an email or phone number so the concierge can reply within one business day."
      );
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/trade", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(buildPayload()),
      });
      const data = (await res.json().catch(() => ({}))) as TradeResponse;
      if (!res.ok || !data.ok || !data.request) {
        setError(
          data.error ?? (zh ? "提交未成功，请稍后再试。" : "The request could not be submitted — please try again.")
        );
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
    setFields({ scenario: form === "enterprise" ? "Enterprise gifting" : "" });
    setInvoice(false);
  }

  if (success) {
    const nextSteps =
      form === "collaboration"
        ? [
            zh ? "礼宾团队核对品牌方向与合作范围" : "The concierge team reviews brand direction and collaboration scope",
            zh ? "初步方案与可行性由人工确认" : "An initial direction and feasibility pass is confirmed by humans",
            zh ? "一个工作日内回复下一步安排" : "You hear back within one business day with next steps",
          ]
        : [
            zh ? "人工礼宾审核数量、预算、周期与地区合规" : "A human concierge reviews quantity, budget, timeline, and region compliance",
            zh ? "确认样品路径与设计方向" : "Sample path and design directions are confirmed",
            zh ? "正式报价单送达后，由你决定是否推进" : "A formal quotation arrives — you decide whether to proceed",
          ];
    return (
      <div className="zx-card zx-fade-up p-6">
        <div className="flex flex-wrap items-center gap-3">
          <StatusPill status={success.status} />
          <p className="font-display text-lg text-porcelain">
            {form === "collaboration"
              ? zh
                ? "联名意向已受理"
                : "Collaboration inquiry received"
              : zh
                ? "询价请求已受理"
                : "Quote request received"}
          </p>
        </div>
        <p className="mt-3 text-sm leading-relaxed text-mist">{success.note}</p>
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
            {zh ? "再提交一份需求" : "Submit another request"}
          </Button>
        </div>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="zx-card p-5 sm:p-6">
      {form === "enterprise" && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label={zh ? "联系人姓名" : "Contact name"}>
            <input value={v("name")} onChange={set("name")} className={inputCls} maxLength={120} placeholder={zh ? "例如：陈伟" : "e.g. Chen Wei"} required />
          </Field>
          <Field label={zh ? "机构 / 公司" : "Organization"}>
            <input value={v("organization")} onChange={set("organization")} className={inputCls} maxLength={200} placeholder={zh ? "公司或机构名称" : "Company or institution"} required />
          </Field>
          <Field label={zh ? "联系方式（邮箱或电话）" : "Contact (email or phone)"}>
            <input value={v("contact")} onChange={set("contact")} className={inputCls} maxLength={200} placeholder="gifting@company.com" required />
          </Field>
          <Field label={zh ? "场景" : "Scenario"}>
            <select value={v("scenario")} onChange={set("scenario")} className={inputCls}>
              {ENTERPRISE_SCENARIOS.map((sc) => (
                <option key={sc.value} value={sc.value}>
                  {zh ? sc.zh : sc.en}
                </option>
              ))}
            </select>
          </Field>
          <Field label={zh ? "数量" : "Quantity"}>
            <input type="number" min={1} value={v("quantity")} onChange={set("quantity")} className={inputCls} placeholder="300" required />
          </Field>
          <Field label={zh ? "预算" : "Budget"}>
            <input value={v("budget")} onChange={set("budget")} className={inputCls} maxLength={120} placeholder={zh ? "例如：450 元 / 件 × 300" : "e.g. 450 RMB / unit × 300"} required />
          </Field>
          <Field label={zh ? "交付期限" : "Deadline"}>
            <input type="date" value={v("deadline")} onChange={set("deadline")} className={inputCls} />
          </Field>
          <Field label={zh ? "交付地区" : "Delivery region"}>
            <input value={v("delivery_region")} onChange={set("delivery_region")} className={inputCls} maxLength={200} placeholder={zh ? "例如：上海 / 北京 / 成都" : "e.g. Shanghai / Beijing / Chengdu"} />
          </Field>
          <div className="sm:col-span-2">
            <label className="flex cursor-pointer items-center gap-3 rounded-lg border border-hairline px-4 py-3">
              <input type="checkbox" checked={invoice} onChange={(e) => setInvoice(e.target.checked)} className="h-4 w-4 accent-gold" />
              <span className="text-sm text-porcelain">{zh ? "需要开具发票" : "Invoice required"}</span>
            </label>
          </div>
          <div className="sm:col-span-2">
            <Field label={zh ? "液体方向" : "Liquid direction"} hint={zh ? "基酒、风味、度数偏好——写方向即可，配方由人工与供应链确认。" : "Base spirit, flavor, proof preference — directions only; formulas are confirmed by humans and the supply chain."}>
              <textarea value={v("liquid_direction")} onChange={set("liquid_direction")} className={`${inputCls} min-h-20`} maxLength={400} placeholder={zh ? "例如：陈年黄酒基底，柑橘皮尾韵" : "e.g. Aged huangjiu base with a citrus-peel finish"} />
            </Field>
          </div>
          <div className="sm:col-span-2">
            <Field label={zh ? "香氛方向" : "Fragrance direction"}>
              <textarea value={v("scent_direction")} onChange={set("scent_direction")} className={`${inputCls} min-h-20`} maxLength={400} placeholder={zh ? "例如：琥珀、桧木、暖香料的室内香氛" : "e.g. Amber, hinoki, warm-spice room mist"} />
            </Field>
          </div>
          <div className="sm:col-span-2">
            <Field label={zh ? "瓶身方向" : "Bottle direction"}>
              <textarea value={v("bottle_direction")} onChange={set("bottle_direction")} className={`${inputCls} min-h-20`} maxLength={400} placeholder={zh ? "例如：宽肩醒酒器造型，青铜瓶盖" : "e.g. Wide-shoulder decanter, bronze cap"} />
            </Field>
          </div>
          <div className="sm:col-span-2">
            <Field label={zh ? "包装方向" : "Packaging direction"}>
              <textarea value={v("packaging_direction")} onChange={set("packaging_direction")} className={`${inputCls} min-h-20`} maxLength={400} placeholder={zh ? "例如：双瓶磁吸礼盒，烫印字母纹样" : "e.g. Two-bottle magnetic case, embossed monogram"} />
            </Field>
          </div>
          <Field label={zh ? "样品路径" : "Sample path"} hint={zh ? "是否需要打样、数量与时间点。" : "Whether you need pre-production samples, how many, and by when."}>
            <input value={v("sample_path")} onChange={set("sample_path")} className={inputCls} maxLength={300} placeholder={zh ? "例如：9 月 15 日前 3 件预产样" : "e.g. 3 pre-production samples by Sept 15"} />
          </Field>
          <Field label={zh ? "物流备注" : "Logistics notes"}>
            <input value={v("logistics_notes")} onChange={set("logistics_notes")} className={inputCls} maxLength={400} placeholder={zh ? "例如：三城分批交付，需温控" : "e.g. Staggered delivery to three cities, temperature-controlled"} />
          </Field>
        </div>
      )}

      {form === "collaboration" && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label={zh ? "品牌 / 机构" : "Brand / organization"}>
            <input value={v("organization")} onChange={set("organization")} className={inputCls} maxLength={200} placeholder={zh ? "品牌、工作室或机构名称" : "Brand, studio, or institution"} required />
          </Field>
          <Field label={zh ? "联系方式（邮箱或电话）" : "Contact (email or phone)"}>
            <input value={v("contact")} onChange={set("contact")} className={inputCls} maxLength={200} placeholder="hello@brand.com" required />
          </Field>
          <div className="sm:col-span-2">
            <Field label={zh ? "联名场景" : "Collaboration scenario"}>
              <input value={v("scenario")} onChange={set("scenario")} className={inputCls} maxLength={200} placeholder={zh ? "例如：设计周城市伴手礼盒 / 文旅联名 / 酒店客房香氛线" : "e.g. Design-week city souvenir box / cultural tourism edition / hotel amenity fragrance line"} required />
            </Field>
          </div>
          <div className="sm:col-span-2">
            <Field label={zh ? "预算范围" : "Budget range"}>
              <input value={v("budget")} onChange={set("budget")} className={inputCls} maxLength={120} placeholder={zh ? "可写“待评估”" : "“To be scoped” is a valid answer"} required />
            </Field>
          </div>
        </div>
      )}

      {form === "concierge" && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label={zh ? "称呼" : "Name"}>
            <input value={v("name")} onChange={set("name")} className={inputCls} maxLength={120} placeholder={zh ? "希望礼宾如何称呼你" : "How the concierge should address you"} required />
          </Field>
          <Field label={zh ? "联系方式（邮箱或电话）" : "Contact (email or phone)"}>
            <input value={v("contact")} onChange={set("contact")} className={inputCls} maxLength={200} placeholder="you@example.com" required />
          </Field>
          <div className="sm:col-span-2">
            <Field label={zh ? "场景" : "Scenario"}>
              <input value={v("scenario")} onChange={set("scenario")} className={inputCls} maxLength={200} placeholder={zh ? "例如：结婚十周年，想为双方父母各定一瓶" : "e.g. Tenth wedding anniversary — one bottle for each set of parents"} required />
            </Field>
          </div>
          <div className="sm:col-span-2">
            <Field label={zh ? "预算" : "Budget"}>
              <input value={v("budget")} onChange={set("budget")} className={inputCls} maxLength={120} placeholder={zh ? "例如：3,000–8,000 元" : "e.g. 3,000–8,000 RMB"} required />
            </Field>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-4">
          <Notice tone="ember">{error}</Notice>
        </div>
      )}

      <div className="mt-5 flex flex-wrap items-center gap-4">
        <Button type="submit" variant="gold" disabled={loading}>
          {loading
            ? zh
              ? "提交中…"
              : "Submitting…"
            : form === "collaboration"
              ? zh
                ? "提交联名意向"
                : "Submit collaboration inquiry"
              : form === "concierge"
                ? zh
                  ? "预约私人礼宾"
                  : "Request a private concierge"
                : zh
                  ? "提交企业定制需求"
                  : "Submit enterprise request"}
        </Button>
        <p className="text-xs leading-relaxed text-mist">
          {zh
            ? "提交后由人工礼宾审核并报价，一个工作日内回复。"
            : "Reviewed and quoted by a human concierge; reply within one business day."}
        </p>
      </div>
    </form>
  );
}
