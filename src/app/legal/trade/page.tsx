import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Notice } from "@/components/ui";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Trade Creative Market Rules",
  description:
    "What may and may not be listed on the ZOTAIX creative market, the authorization workflow, review states, fees, and dispute handling. No user-to-user alcohol resale.",
  path: "/legal/trade",
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

export default async function TradeRulesPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const doc = db().legal_docs.find((d) => d.slug === "trade");

  return (
    <article className="space-y-10">
      <header>
        <h1 className="font-display text-3xl text-porcelain sm:text-4xl">
          {zh ? "创意市场规则" : "Trade Creative Market Rules"}
        </h1>
        <p className="mt-3 text-xs uppercase tracking-[0.2em] text-gold">
          {zh ? "版本" : "Version"} {doc?.version ?? "1.0"} · {zh ? "生效日期" : "Effective date"}{" "}
          {doc?.effective_date ?? "2026-05-01"}
        </p>
        <p className="mt-4 text-sm leading-relaxed text-mist">
          {zh
            ? "创意市场交易的是创意与权利，而不是酒。本规则界定可上架的内容、禁止事项、授权流程与争议处理。"
            : "The creative market trades in creativity and rights — never in bottles. These rules define what may be listed, what is prohibited, how authorization works, and how disputes are handled."}
        </p>
      </header>

      <Sec n={1} title={zh ? "市场的性质" : "What the market is"}>
        <p>
          The trade market is where reviewed creative value changes hands: proposals, design concepts, project rights,
          and collaboration opportunities. Every listing passes human review before it appears, every transaction runs
          through the platform's authorization workflow, and physical alcohol never trades between users. Physical
          production, where a traded right leads to one, always re-enters the standard quotation and compliance path.
        </p>
      </Sec>

      <Sec n={2} title={zh ? "允许上架的内容" : "Allowed listings"}>
        <ul className="list-disc space-y-1.5 pl-5">
          <li><span className="text-porcelain">Public proposals</span> — AI-generated concepts their owner has chosen to make public and offer for licensed reuse;</li>
          <li><span className="text-porcelain">Label and bottle concepts</span> — visual and copy directions, licensed for reuse or transferred;</li>
          <li><span className="text-porcelain">Founder rights</span> — founder identity in a co-creation project, transferred through reviewed authorization;</li>
          <li><span className="text-porcelain">Participation slots</span> — a reserved place in a gathering project, transferred before production lock;</li>
          <li><span className="text-porcelain">Digital badges</span> — digital marks whose issuing terms permit transfer;</li>
          <li><span className="text-porcelain">Public archive pages</span> — sponsorship or featuring of public Reserve pages;</li>
          <li><span className="text-porcelain">Collaboration applications</span> — offers to collaborate on a project or concept;</li>
          <li><span className="text-porcelain">Enterprise leads</span> — introductions of enterprise gifting demand, handled through the concierge;</li>
          <li><span className="text-porcelain">Designer copyright income logic</span> — reviewed arrangements under which a designer earns a share when their listed concept is produced, as recorded in the authorization agreement.</li>
        </ul>
      </Sec>

      <Sec n={3} title={zh ? "禁止事项" : "Prohibited"}>
        <Notice tone="ember" title={zh ? "红线" : "Hard lines"}>
          {zh
            ? "以下行为一律禁止，且会触发审核升级：用户间酒类转售；未经审核的兑换券；无资质销售者；引导站外交易；未经审核的现货交易。"
            : "The following are prohibited without exception and trigger escalated review: user-to-user alcohol resale; unreviewed vouchers; unauthorized sellers; guidance toward off-platform transactions; unreviewed spot trading."}
        </Notice>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>
            <span className="text-porcelain">User-to-user alcohol resale.</span> Bottles, batches, and any physical
            alcohol may not be sold, auctioned, or exchanged between users through or around the market — alcohol
            distribution requires licensed channels and the platform's compliance checks.
          </li>
          <li>
            <span className="text-porcelain">Unreviewed vouchers.</span> Coupons, redemption codes, or claims on
            products may not be created or listed unless issued and reviewed by the platform.
          </li>
          <li>
            <span className="text-porcelain">Unauthorized sellers.</span> Listing on behalf of a business or as a
            reseller requires verified qualification through the authorization workflow.
          </li>
          <li>
            <span className="text-porcelain">External transaction guidance.</span> Using listings, messages, or archive
            pages to move payment or delivery outside the platform's reviewed channels.
          </li>
          <li>
            <span className="text-porcelain">Unreviewed spot trading.</span> Immediate-settlement trading of any item
            that has not passed listing review, including bulk or speculative flipping of participation slots.
          </li>
        </ul>
      </Sec>

      <Sec n={4} title={zh ? "授权流程" : "Authorization workflow"}>
        <p>Every transaction follows the same reviewed path:</p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>the seller submits an authorization request describing the item, scope (license or transfer), territory, duration, and price expectation;</li>
          <li>the platform verifies ownership or founder identity, checks the item's review history, and confirms the item's trade eligibility;</li>
          <li>compliance and human review states are recorded on the request (see Section 5);</li>
          <li>on approval, the parties receive a written authorization agreement stating the exact rights conveyed; the transfer takes effect when both parties accept;</li>
          <li>the outcome is recorded against the item — founder transfers update the project record, badge transfers update the Reserve record.</li>
        </ul>
      </Sec>

      <Sec n={5} title={zh ? "审核状态" : "Review states"}>
        <p>
          Requests and listings carry two visible states. <span className="text-porcelain">Compliance status</span> —
          unchecked, passed, or flagged — reflects the automated and manual compliance screening.{" "}
          <span className="text-porcelain">Human review status</span> — pending, approved, rejected, revision, or
          escalated — reflects the concierge team's decision. Nothing trades while either state is unresolved; a
          flagged or escalated item is frozen until review completes, and rejected items may be revised and
          resubmitted with the reasons addressed.
        </p>
      </Sec>

      <Sec n={6} title={zh ? "费用" : "Fees"}>
        <p>
          Market fees are set per agreement: the authorization agreement for each transaction states any platform
          service fee, the designer income share where one applies, and who bears payment costs. There is no hidden or
          default percentage — if a fee is not written in your agreement, it is not owed. Enterprise-scale
          arrangements are quoted individually by the concierge.
        </p>
      </Sec>

      <Sec n={7} title={zh ? "争议处理" : "Dispute handling"}>
        <p>
          Disputes between parties to a market transaction are raised first with the concierge at
          concierge@zotaix.example, citing the authorization agreement reference. The platform reviews the recorded
          request, the agreement, and both parties' statements, and issues a resolution within the review workflow —
          which may include reversing a transfer, releasing a held payment, or restoring a right. Where a party rejects
          the platform's resolution, the governing-law and dispute provisions of the User Terms apply. Fraud,
          impersonation, and repeated prohibited conduct lead to market exclusion and, where warranted, account
          termination.
        </p>
      </Sec>

      <Sec n={8} title={zh ? "联系" : "Contact"}>
        <p>
          Questions about listing eligibility, authorization scope, or seller qualification:
          concierge@zotaix.example, or the channels on the Contact page.
        </p>
      </Sec>
    </article>
  );
}
