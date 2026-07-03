import type { Metadata } from "next";
import type { ReactNode } from "react";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "User Terms",
  description:
    "The user terms of the ZOTAIX platform: account tiers, acceptable use, AI content ownership and licensing, co-creation participation, order and quotation process, termination, and liability.",
  path: "/legal/terms",
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

export default async function TermsPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const doc = db().legal_docs.find((d) => d.slug === "terms");

  return (
    <article className="space-y-10">
      <header>
        <h1 className="font-display text-3xl text-porcelain sm:text-4xl">{zh ? "用户协议" : "User Terms"}</h1>
        <p className="mt-3 text-xs uppercase tracking-[0.2em] text-gold">
          {zh ? "版本" : "Version"} {doc?.version ?? "1.0"} · {zh ? "生效日期" : "Effective date"}{" "}
          {doc?.effective_date ?? "2026-05-01"}
        </p>
        <p className="mt-4 text-sm leading-relaxed text-mist">
          {zh
            ? "本协议约定你与 ZOTAIX 卓序平台运营方之间关于使用本平台的权利与义务。使用平台即表示你已阅读并同意本协议。"
            : "These terms govern the relationship between you and the operator of the ZOTAIX platform. By using the platform you confirm that you have read and accept them."}
        </p>
      </header>

      <Sec n={1} title={zh ? "服务概述" : "What the service is"}>
        <p>
          ZOTAIX is an AI concierge customization platform. It turns emotions, relationships, scenarios, and budgets
          into creative objects: liquid directions, fragrance directions, bottle and label design directions, packaging
          copy, gifting concepts, and digital identity records. The platform's first principle is that you create and
          save a personalized object before deciding whether it becomes a physical product. The platform never places
          anything in a cart on your behalf, and saving an object creates no payment obligation.
        </p>
        <p>
          Physical production — small-batch castings, premium bespoke gifts, co-creation runs, and enterprise gifting
          programs — always passes through human concierge confirmation, supply-chain confirmation, compliance checks,
          and a written quotation before any commitment exists (see Section 7).
        </p>
      </Sec>

      <Sec n={2} title={zh ? "账户层级" : "Account tiers"}>
        <p>The platform distinguishes four tiers of access:</p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>
            <span className="text-porcelain">Guest.</span> You may use the daily concierge within a visitor quota
            without an account. Guest conversations are tracked only by an anonymous visitor identifier and cannot be
            saved to an archive.
          </li>
          <li>
            <span className="text-porcelain">Registered user.</span> A free account lets you save object drafts, keep
            conversations, maintain a profile with optional self-expression tags, join co-creation projects, and hold
            Reserve records, within the free daily quota.
          </li>
          <li>
            <span className="text-porcelain">Member (Core Sequence).</span> Paid membership plans raise daily and
            monthly generation quotas and unlock benefits such as export, Reserve certificates, co-creation founding,
            and concierge priority, as defined in the Membership Service Agreement.
          </li>
          <li>
            <span className="text-porcelain">Enterprise.</span> Enterprise accounts access the Maison line: gifting
            programs, brand collaborations, sample paths, invoicing, and dedicated human concierge management, governed
            additionally by individual written agreements.
          </li>
        </ul>
        <p>
          You are responsible for the accuracy of your registration data and for keeping your credentials confidential.
          Accounts are personal and may not be transferred or shared. Accounts may only be created by adults; see the
          Minor Protection Notice.
        </p>
      </Sec>

      <Sec n={3} title={zh ? "可接受使用" : "Acceptable use"}>
        <p>You agree not to use the platform to:</p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>generate, publish, or distribute unlawful, hateful, harassing, defamatory, or sexually exploitative content;</li>
          <li>promote alcohol to minors, or circumvent the age gate or any regional restriction;</li>
          <li>infringe third-party intellectual property, personality rights, or trade secrets in briefs, names, label copy, or uploaded material;</li>
          <li>misrepresent AI proposals as confirmed products, medical advice, or professional formulation guidance;</li>
          <li>scrape, probe, overload, or reverse engineer the service, or resell access to it;</li>
          <li>manipulate quotas, votes, co-creation thresholds, or moderation outcomes through automation or coordinated fake accounts.</li>
        </ul>
        <p>
          The operator may remove content, restrict features, or suspend accounts that violate this section, applying
          the review states described in the moderation workflow (approved, rejected, revision, escalated).
        </p>
      </Sec>

      <Sec n={4} title={zh ? "AI 内容的归属与授权" : "AI content: ownership and license"}>
        <p>
          <span className="text-porcelain">Your briefs are yours.</span> Everything you type into the concierge — your
          emotions, scenarios, recipients, budgets, and creative direction — remains your material. You grant the
          operator a non-exclusive license to process it for the purpose of operating the service, as described in the
          Privacy Policy.
        </p>
        <p>
          <span className="text-porcelain">Generated proposals are licensed to you.</span> Structured proposals created
          for you (liquid directions, fragrance directions, names, label copy, visual directions, digital marks) are
          licensed to you for personal use, gifting, publication to your own channels, and — where the object passes
          review — co-creation and trade listing. When a proposal becomes a physical order, the accompanying quotation
          states the scope of commercial rights for that production run.
        </p>
        <p>
          <span className="text-porcelain">The platform retains the right to review.</span> All generated and
          user-submitted content that is published, listed, or ordered is subject to human review across the platform's
          moderation dimensions (sensitive content, alcohol compliance, minor safety, copyright, feasibility, public
          display, trade eligibility). Review may delay, condition, or decline publication or production.
        </p>
      </Sec>

      <Sec n={5} title={zh ? "共创参与" : "Co-creation participation"}>
        <p>
          Publishing a concept to the co-creation pool, joining a project, or voting is governed by the Co-Creation
          Pool Rules, which form part of these terms. In summary: thresholds unlock stages of collective customization;
          reservations are held, not charged, until a project passes threshold and review; and reservations are released
          in full if either fails. Founder and participant benefits are stated on each project page before you join.
        </p>
      </Sec>

      <Sec n={6} title={zh ? "订单与报价流程" : "Orders, quotations, and human confirmation"}>
        <p>
          No AI output constitutes an offer, order, or contract. The path from saved object to physical delivery is
          always: (a) you request a quote or casting; (b) a human concierge reviews quantity, budget, deadline,
          delivery region, and compliance; (c) supply-chain feasibility is confirmed; (d) you receive a written
          quotation stating the final product scope, price, timeline, logistics, packaging, and aftercare; (e) a
          contract exists only when you accept that quotation. Where payment channels are operated in concierge-confirmed
          mode, orders are recorded and a human concierge confirms payment through the channels stated in the quotation.
        </p>
        <p>
          Physical alcohol deliveries are additionally subject to the Alcohol Compliance Notice, including age, region,
          logistics, and qualification checks.
        </p>
      </Sec>

      <Sec n={7} title={zh ? "终止" : "Termination"}>
        <p>
          You may close your account at any time from your profile or by writing to the concierge; the Privacy Policy
          describes what is deleted and what may be retained. The operator may suspend or terminate accounts for
          material or repeated violations of these terms, for legal compliance, or where required by authorities, and
          will state the reason unless legally prevented. Accepted quotations in progress survive termination to the
          extent needed to complete or wind down the transaction.
        </p>
      </Sec>

      <Sec n={8} title={zh ? "免责与责任限制" : "Disclaimers and limitation of liability"}>
        <p>
          The creative service is provided as described, without warranty that any proposal will suit your taste, be
          producible at a given price, or achieve a particular emotional effect. AI proposals are creative guidance,
          not professional, medical, or regulatory advice. To the maximum extent permitted by applicable law, the
          operator's aggregate liability for claims arising from the free creative service is limited to the amount you
          paid for the platform services in the twelve months preceding the claim; for physical orders, liability is
          governed by the accepted quotation and mandatory consumer protection law. Nothing in these terms limits
          liability for intent, gross negligence, or harm to life and health where such limits are not permitted.
        </p>
      </Sec>

      <Sec n={9} title={zh ? "适用法律" : "Governing law"}>
        <p>
          These terms are governed by the laws applicable at the operator's registered domicile, without prejudice to
          mandatory consumer protections of the jurisdiction in which you habitually reside. Disputes are first raised
          with the concierge; where escalation is needed, the competent courts or dispute-resolution bodies at the
          operator's registered domicile apply, unless mandatory law designates otherwise.
        </p>
      </Sec>

      <Sec n={10} title={zh ? "协议变更与联系" : "Changes and contact"}>
        <p>
          The operator may update these terms; the version and effective date at the top of this page always identify
          the current text, and material changes are announced in the app and on this page before they take effect.
          Continued use after the effective date constitutes acceptance. Questions about these terms:
          concierge@zotaix.example, or the channels listed on the Contact page.
        </p>
      </Sec>
    </article>
  );
}
