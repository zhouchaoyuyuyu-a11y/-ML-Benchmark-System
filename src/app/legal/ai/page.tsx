import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Notice } from "@/components/ui";
import { complianceNotice } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "AI Generated Content Notice",
  description:
    "What AI-generated content on ZOTAIX is and is not: creative guidance, not final formulas or professional advice. Quota tiers, human confirmation, provider fallback, and moderation.",
  path: "/legal/ai",
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

export default async function AiNoticePage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const doc = data.legal_docs.find((d) => d.slug === "ai");
  const s = data.settings;

  return (
    <article className="space-y-10">
      <header>
        <h1 className="font-display text-3xl text-porcelain sm:text-4xl">
          {zh ? "AI 生成内容声明" : "AI Generated Content Notice"}
        </h1>
        <p className="mt-3 text-xs uppercase tracking-[0.2em] text-gold">
          {zh ? "版本" : "Version"} {doc?.version ?? "1.0"} · {zh ? "生效日期" : "Effective date"}{" "}
          {doc?.effective_date ?? "2026-05-01"}
        </p>
        <p className="mt-4 text-sm leading-relaxed text-mist">
          {zh
            ? "本声明界定 ZOTAIX 上 AI 生成内容的性质、边界与审核方式。它是所有创意提案的前提。"
            : "This notice defines what AI-generated content on ZOTAIX is, where its boundaries lie, and how it is reviewed. It is the premise behind every creative proposal."}
        </p>
      </header>

      <Sec n={1} title={zh ? "AI 输出的性质" : "The nature of AI output"}>
        <p>
          Everything the concierge generates — liquid directions, fragrance directions, bottle and label directions,
          names, label copy, gift stories, digital marks — is <span className="text-porcelain">creative guidance</span>.
          It expresses a direction and a mood; it is not a final production formula, a recipe, a technical
          specification, or a promise that a specific product can be made at a specific price.
        </p>
        <p>
          AI output is also not medical, psychological, nutritional, legal, or other professional advice. The daily
          concierge responds to how your day feels, but it does not diagnose, treat, or replace professional care. If
          you are struggling, please reach out to a qualified professional or a local support service.
        </p>
        <Notice tone="gold">{zh ? complianceNotice.zh : complianceNotice.en}</Notice>
      </Sec>

      <Sec n={2} title={zh ? "配额层级" : "Quota tiers"}>
        <p>
          Generation is metered so the service stays responsive and fair. Current quotas, as configured in platform
          settings and shown at purchase:
        </p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>
            <span className="text-porcelain">Guest:</span> {s.guest_daily_chat} concierge conversations per day,
            counted by an anonymous visitor identifier.
          </li>
          <li>
            <span className="text-porcelain">Registered (free):</span> {s.free_daily_chat} conversations per day, plus
            saving drafts and profiles.
          </li>
          <li>
            <span className="text-porcelain">Core Sequence Lite:</span> {s.lite_daily_chat} conversations per day and{" "}
            {s.lite_monthly_proposals} structured proposals per month.
          </li>
          <li>
            <span className="text-porcelain">Core Sequence Pro:</span> {s.pro_daily_chat} conversations per day and{" "}
            {s.pro_monthly_proposals} structured proposals per month, with creative and export features.
          </li>
        </ul>
        <p>
          When a quota is reached, the platform says so plainly and shows the applicable upgrade path; it never
          silently degrades your results. Quota definitions and billing are governed by the Membership Service
          Agreement.
        </p>
      </Sec>

      <Sec n={3} title={zh ? "人工确认" : "Human confirmation"}>
        <p>
          Premium customization and any physical delivery require human confirmation. An AI proposal becomes a product
          only after a human concierge has reviewed your request, the supply chain has confirmed feasibility, and you
          have accepted a written quotation. Final pricing, timeline, production scope, logistics, packaging, and
          aftercare are all determined by that manual quotation — never by the AI output itself. For alcohol, the
          checks in the Alcohol Compliance Notice apply in addition.
        </p>
      </Sec>

      <Sec n={4} title={zh ? "生成引擎与回退机制" : "Generation engines and fallback"}>
        <p>
          The platform can operate with an external AI provider or on its own{" "}
          <span className="text-porcelain">deterministic atelier engine</span>. Where the operator has configured an
          external provider, briefs are processed by that provider under the terms of the Privacy Policy. Where no
          external provider is configured, the atelier engine composes proposals from the platform's curated libraries
          of liquid, scent, and visual directions — deterministically, so the same brief yields consistent, reviewable
          results. Both engines produce the same structured proposal format, and every response indicates which engine
          answered. This dual-engine design is how the concierge remains available at all times.
        </p>
      </Sec>

      <Sec n={5} title={zh ? "生成内容的审核" : "Moderation of generated content"}>
        <p>
          Generated content that stays in a private conversation is yours to keep or discard. The moment content is
          published, listed, ordered, or shared through platform features, it enters human review across these
          dimensions:
        </p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>sensitive content (hate, harassment, explicit material, dangerous instructions);</li>
          <li>alcohol compliance (no promotion to minors, no health claims, no regulatory violations);</li>
          <li>minor safety;</li>
          <li>copyright and third-party rights;</li>
          <li>feasibility (no promises the supply chain cannot keep);</li>
          <li>public display suitability;</li>
          <li>trade eligibility for market listings.</li>
        </ul>
        <p>
          Review outcomes are approved, rejected, revision requested, or escalated. The platform may decline to
          generate, publish, or produce content that fails these dimensions, as set out in the User Terms.
        </p>
      </Sec>

      <Sec n={6} title={zh ? "准确性与偏差" : "Accuracy and limitations"}>
        <p>
          AI systems can be wrong, generic, or oddly confident. Proposals may reference styles, notes, or techniques
          imprecisely; names may resemble existing marks and are checked during review, not at generation time. Treat
          every proposal as a first draft from a talented but unverified collaborator: keep what resonates, question
          what seems off, and rely on the human quotation stage for anything factual, financial, or physical.
        </p>
      </Sec>

      <Sec n={7} title={zh ? "归属与反馈" : "Ownership and reporting"}>
        <p>
          Ownership and licensing of briefs and proposals are defined in the User Terms: your briefs remain yours, and
          generated proposals are licensed to you. If a generated result is offensive, unsafe, or appears to infringe
          rights, report it to concierge@zotaix.example with the conversation or object reference; reports enter the
          moderation queue and receive a reply. Questions about this notice: the channels on the Contact page.
        </p>
      </Sec>
    </article>
  );
}
