import type { Metadata } from "next";
import type { ReactNode } from "react";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Membership Service Agreement",
  description:
    "The Core Sequence membership terms: plans and pricing, billing cycles, Order Energy quota definitions, refunds and cancellation, benefit changes, and founder identity.",
  path: "/legal/membership",
});

function Sec({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <section className="space-y-3">
      <h2 className="font-display text-lg text-porcelain">
        {n}. {title}
      </h2>
      <div className="space-y-3 text-sm leading-relaxed text-mist">{children}</div>
    </section>
  );
}

export default async function MembershipAgreementPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const doc = data.legal_docs.find((d) => d.slug === "membership");
  const s = data.settings;

  return (
    <article className="space-y-10">
      <header>
        <h1 className="font-display text-3xl text-porcelain sm:text-4xl">
          {zh ? "会员服务协议" : "Membership Service Agreement"}
        </h1>
        <p className="mt-3 text-xs uppercase tracking-[0.2em] text-gold">
          {zh ? "版本" : "Version"} {doc?.version ?? "1.0"} · {zh ? "生效日期" : "Effective date"}{" "}
          {doc?.effective_date ?? "2026-05-01"}
        </p>
        <p className="mt-4 text-sm leading-relaxed text-mist">
          {zh
            ? "本协议约定核心序列（Core Sequence）会员服务的开通、计费、配额、退款与权益变更规则，是用户协议的组成部分。"
            : "This agreement governs the Core Sequence membership service — activation, billing, quotas, refunds, and benefit changes. It forms part of the User Terms."}
        </p>
      </header>

      <Sec n={1} title={zh ? "核心序列是什么" : "What Core Sequence is"}>
        <p>
          Core Sequence is the platform's membership program. Its vocabulary is deliberately its own: members are Order
          Builders, usage quota is <span className="text-porcelain">Order Energy</span>, and redemption into physical
          objects is Physical Casting. The playful names change nothing legally — wherever this agreement uses a plain
          term and a page uses the Order World term, they mean the same thing, and this agreement's definitions
          control.
        </p>
      </Sec>

      <Sec n={2} title={zh ? "方案与定价" : "Plans and pricing"}>
        <p>
          Plans and prices are set in platform settings and always shown to you at the moment of purchase; the purchase
          screen is authoritative for what you pay. As currently configured:
        </p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>
            <span className="text-porcelain">Core Sequence Lite</span> — ¥{s.lite_price_month}/month or ¥
            {s.lite_price_quarter}/quarter: {s.lite_daily_chat} concierge conversations per day,{" "}
            {s.lite_monthly_proposals} structured proposals per month, Reserve records enabled.
          </li>
          <li>
            <span className="text-porcelain">Core Sequence Pro</span> — ¥{s.pro_price_month}/month or ¥
            {s.pro_price_quarter}/quarter: {s.pro_daily_chat} conversations per day, {s.pro_monthly_proposals}{" "}
            proposals per month, export, Reserve certificates, image generation allowance, and concierge priority.
          </li>
        </ul>
        <p>
          Enterprise access is quoted individually and governed by written agreement; it is not purchased through this
          page.
        </p>
      </Sec>

      <Sec n={3} title={zh ? "计费周期" : "Billing cycles"}>
        <p>
          Memberships run in monthly or quarterly cycles from the day of activation. A cycle is a fixed prepaid period:
          it does not auto-extend without a renewal, and you are shown the renewal price before any renewal charge.
          Switching plans takes effect at the start of the next cycle unless you upgrade mid-cycle, in which case the
          upgrade is applied immediately and the price difference for the remaining period is included in the upgrade
          quotation shown to you.
        </p>
      </Sec>

      <Sec n={4} title={zh ? "配额定义（秩序能量）" : "Quota definitions (Order Energy)"}>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>
            <span className="text-porcelain">Daily conversations</span> reset at midnight (server time) each day.
            Unused daily conversations do not roll over.
          </li>
          <li>
            <span className="text-porcelain">Monthly structured proposals</span> count each premium generation
            (spirit, fragrance, style, co-create, enterprise modes) against the monthly allowance, which resets on your
            cycle date. Unused proposals do not roll over.
          </li>
          <li>
            <span className="text-porcelain">Image and export allowances</span> apply per plan as shown at purchase.
          </li>
          <li>
            Quota is consumed only by successful generations; failed requests are not counted. When quota runs out, the
            platform states it plainly and shows the upgrade path — results are never silently degraded.
          </li>
        </ul>
      </Sec>

      <Sec n={5} title={zh ? "退款与取消" : "Refunds and cancellation"}>
        <p>
          You can cancel renewal at any time; your benefits continue to the end of the paid cycle. For a first
          purchase, if you have consumed no membership quota and request a refund within seven days of activation, the
          cycle is refunded in full. After quota has been consumed, refunds for the remaining period are handled
          case-by-case by the concierge, and always granted where the platform failed to provide the contracted
          benefits or where mandatory consumer law grants a withdrawal right. Refunds return through the original
          payment channel; where the order was concierge-confirmed, the concierge arranges the return transfer.
        </p>
      </Sec>

      <Sec n={6} title={zh ? "权益变更" : "Benefit changes"}>
        <p>
          The operator may adjust plan contents and prices in platform settings. Changes never reduce what you already
          paid for: adjustments take effect for you at your next cycle at the earliest, are announced in the app and on
          this page before they apply, and if a change materially reduces the plan you are on, you may cancel with a
          pro-rated refund of the remaining period. Improvements (higher quotas, added benefits) may be applied to
          running cycles immediately.
        </p>
      </Sec>

      <Sec n={7} title={zh ? "支付方式与礼宾确认" : "Payment channels and concierge confirmation"}>
        <p>
          Purchases run through the payment channels configured by the operator (such as WeChat Pay, Alipay, Stripe,
          PayPal, or bank transfer). Where a channel is operated in concierge-confirmed mode, the platform records your
          order in a confirmation state and a human concierge completes the payment arrangement with you through the
          stated channel — your membership activates upon that confirmation. Orders recorded in test mode are marked as
          such, carry no charge, and create no payment obligation on either side.
        </p>
      </Sec>

      <Sec n={8} title={zh ? "发起人身份" : "Founder identity"}>
        <p>
          Membership can carry founder identity in co-creation: members may found projects in the co-creation pool, and
          founder benefits (founder serials, engravings, archive pages, colophon credits) attach to the founding
          account personally. Founder identity is not transferable except through a reviewed authorization in the trade
          market, survives a lapse in membership for projects already founded, and is governed in detail by the
          Co-Creation Pool Rules.
        </p>
      </Sec>

      <Sec n={9} title={zh ? "终止与联系" : "Termination and contact"}>
        <p>
          Closing your account ends membership; Section 5 governs any refund of the running cycle. Membership may be
          suspended for violations under the User Terms, with quota paused during suspension and restored if the
          suspension is lifted. Questions about billing, quotas, or this agreement: concierge@zotaix.example, or the
          channels on the Contact page.
        </p>
      </Sec>
    </article>
  );
}
