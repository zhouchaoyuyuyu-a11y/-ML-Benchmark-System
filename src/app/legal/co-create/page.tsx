import type { Metadata } from "next";
import type { ReactNode } from "react";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Co-Creation Pool Rules",
  description:
    "The rules of the ZOTAIX co-creation pool: publish flow, quantity thresholds and what each unlocks, founder and participant benefits, review dimensions, refunds, and IP.",
  path: "/legal/co-create",
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

export default async function CoCreateRulesPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const doc = data.legal_docs.find((d) => d.slug === "co-create");
  const s = data.settings;

  const thresholds: { count: number; meaning: string }[] = [
    {
      count: s.co_create_public_threshold,
      meaning:
        "Public page. Ten supporters open the project's public co-creation page: the concept, tags, progress bar, and voting become visible to everyone.",
    },
    {
      count: s.co_create_review_threshold,
      meaning:
        "Formal review. Thirty reservations move the project into the full compliance and feasibility review queue, with atelier input on producibility.",
    },
    {
      count: s.co_create_label_threshold,
      meaning:
        "Label round. Fifty reservations unlock collective label and gift-box theming: participants vote on label variants and packaging directions.",
    },
    {
      count: s.co_create_flavor_threshold,
      meaning:
        "Flavor review. One hundred bottles unlock flavor-direction review with the atelier — the stage where the liquid direction is matched to a producible batch profile.",
    },
    {
      count: s.co_create_enterprise_threshold,
      meaning:
        "Enterprise scale. Three hundred units bring enterprise-grade gifting review, staggered logistics planning, and invoicing support.",
    },
    {
      count: s.co_create_supply_threshold,
      meaning:
        "Dedicated run. Five hundred units qualify the project for a dedicated supply run with extended aftercare and a replenishment window in Reserve.",
    },
    {
      count: s.co_create_partner_threshold,
      meaning:
        "Partner series. One thousand units open partner-level collaboration: a named series, long-term supply planning, and a revenue-share discussion with the founder under a written agreement.",
    },
  ];

  return (
    <article className="space-y-10">
      <header>
        <h1 className="font-display text-3xl text-porcelain sm:text-4xl">
          {zh ? "共创池规则" : "Co-Creation Pool Rules"}
        </h1>
        <p className="mt-3 text-xs uppercase tracking-[0.2em] text-gold">
          {zh ? "版本" : "Version"} {doc?.version ?? "1.0"} · {zh ? "生效日期" : "Effective date"}{" "}
          {doc?.effective_date ?? "2026-05-01"}
        </p>
        <p className="mt-4 text-sm leading-relaxed text-mist">
          {zh
            ? "共创池让一个人的草稿变成一群人的批次。本规则说明发布流程、人数阈值、发起人与参与者的权益、评审维度与退款处理。"
            : "The co-creation pool turns one person's draft into a group's batch. These rules define the publish flow, the thresholds, founder and participant benefits, review dimensions, and refund handling."}
        </p>
      </header>

      <Sec n={1} title={zh ? "发布流程" : "Publish flow"}>
        <p>
          Any member can found a project from a saved draft or a fresh concept. The flow is: (a) the founder submits
          title, concept, product type, target quantity, and emotion tags; (b) the project enters pre-publication
          review across the dimensions in Section 4; (c) an approved project appears in the pool in gathering state,
          where supporters can vote and reserve; (d) crossing each quantity threshold unlocks the corresponding stage
          in Section 2; (e) a project that reaches its target and passes final review moves to production, then
          delivery, with every participant's unit bound to the shared batch archive in Reserve.
        </p>
        <p>
          Founding requires an active membership; joining requires a registered account; voting is open to everyone.
          Reservations made while gathering are held commitments, not charges — see Section 5.
        </p>
      </Sec>

      <Sec n={2} title={zh ? "人数阈值与含义" : "Thresholds and what they unlock"}>
        <div className="space-y-2">
          {thresholds.map((t) => (
            <div key={t.count} className="flex gap-4 rounded-lg border border-hairline bg-obsidian/40 px-4 py-3">
              <span className="font-display shrink-0 text-lg text-gold">{t.count}</span>
              <p className="text-sm leading-relaxed text-mist">{t.meaning}</p>
            </div>
          ))}
        </div>
        <p>
          Threshold values are set in platform settings and shown on every project page; the values above are those
          currently in force. A threshold reached is only ever a gate to the next review — it never bypasses one.
        </p>
      </Sec>

      <Sec n={3} title={zh ? "发起人与参与者权益" : "Founder and participant benefits"}>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>
            <span className="text-porcelain">Founders</span> receive the founder benefit stated on the project page
            before anyone joins — typically a founder-edition serial, an engraving or colophon credit, and an exclusive
            archive page — plus naming influence during the label and flavor rounds and first access to the partner
            discussion at the highest threshold. Founder identity attaches to the founding account and is governed by
            the Membership Service Agreement.
          </li>
          <li>
            <span className="text-porcelain">Participants</span> receive the units they reserved, participation in the
            votes their threshold stage unlocks, a participant digital mark, and a Reserve record binding their unit to
            the shared batch archive.
          </li>
        </ul>
        <p>
          Benefits are stated per project and cannot be reduced after the first reservation; they may be added to.
        </p>
      </Sec>

      <Sec n={4} title={zh ? "评审维度" : "Review dimensions"}>
        <p>Every project is reviewed, at publication and again before production, across seven dimensions:</p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li><span className="text-porcelain">Sensitive content</span> — no hate, harassment, explicit, or dangerous material;</li>
          <li><span className="text-porcelain">Alcohol compliance</span> — the Alcohol Compliance Notice applies to every alcoholic project;</li>
          <li><span className="text-porcelain">Minor safety</span> — a dedicated check on concept, imagery, framing, and audience;</li>
          <li><span className="text-porcelain">Copyright</span> — names, artwork, and copy must not infringe third-party rights;</li>
          <li><span className="text-porcelain">Feasibility</span> — the atelier confirms the concept can actually be produced as described;</li>
          <li><span className="text-porcelain">Public display</span> — suitability of the public page's content;</li>
          <li><span className="text-porcelain">Trade eligibility</span> — whether project rights may be listed on the trade market.</li>
        </ul>
        <p>
          Outcomes are approved, rejected, revision requested, or escalated, with reviewer notes retained in the
          moderation log. A revision request pauses gathering until the founder resubmits.
        </p>
      </Sec>

      <Sec n={5} title={zh ? "失败与退款处理" : "Failure and refund handling"}>
        <p>
          Reservations are held, not charged, while a project gathers. If a project{" "}
          <span className="text-porcelain">does not reach its target quantity</span> within its gathering window, or{" "}
          <span className="text-porcelain">fails any review</span> at publication or pre-production, all reservations
          are released in full: held commitments are cancelled, any amounts already collected for that project are
          refunded through their original channel, and every participant is notified with the reason. Founders whose
          projects fail review may revise and resubmit; supporters must actively re-reserve — reservations never carry
          over silently to a changed project.
        </p>
      </Sec>

      <Sec n={6} title={zh ? "共创概念的知识产权" : "IP of co-created concepts"}>
        <p>
          The founder retains authorship of the founding concept. By publishing, the founder grants the platform and
          the project's participants the license needed to run the project: display on the public page, collective
          voting on variants, production of the batch, and archive pages in Reserve. Contributions selected in label or
          flavor rounds are licensed by their contributors to the project on the same basis. Commercial rights beyond
          the batch — reissues, named series, external sales — remain with the founder and are exercised through the
          trade market's authorization workflow or a written partner agreement. The platform does not take ownership
          of co-created concepts.
        </p>
      </Sec>

      <Sec n={7} title={zh ? "行为与联系" : "Conduct and contact"}>
        <p>
          Vote manipulation, coordinated fake reservations, and harassment of founders or participants violate the
          User Terms and lead to removal from projects. Questions and disputes about a project:
          concierge@zotaix.example with the project reference, or the channels on the Contact page.
        </p>
      </Sec>
    </article>
  );
}
