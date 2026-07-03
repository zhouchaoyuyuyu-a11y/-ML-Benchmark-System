import type { Metadata } from "next";
import type { ReactNode } from "react";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Minor Protection Notice",
  description:
    "How ZOTAIX protects minors: adult-only accounts, the alcohol age gate, zero-proof product policy, no marketing to minors, co-creation minor-safety review, and the reporting channel.",
  path: "/legal/minors",
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

export default async function MinorsPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const doc = db().legal_docs.find((d) => d.slug === "minors");

  return (
    <article className="space-y-10">
      <header>
        <h1 className="font-display text-3xl text-porcelain sm:text-4xl">
          {zh ? "未成年人保护声明" : "Minor Protection Notice"}
        </h1>
        <p className="mt-3 text-xs uppercase tracking-[0.2em] text-gold">
          {zh ? "版本" : "Version"} {doc?.version ?? "1.0"} · {zh ? "生效日期" : "Effective date"}{" "}
          {doc?.effective_date ?? "2026-05-01"}
        </p>
        <p className="mt-4 text-sm leading-relaxed text-mist">
          {zh
            ? "一个涉及酒类的平台对未成年人负有明确责任。本声明说明 ZOTAIX 如何履行这份责任。"
            : "A platform that touches alcohol carries a clear responsibility toward minors. This notice states how ZOTAIX carries it."}
        </p>
      </header>

      <Sec n={1} title={zh ? "基本立场" : "Our position"}>
        <p>
          ZOTAIX is a platform for adults. Minors may not hold accounts, may not order, may not join co-creation
          projects, and are never a marketing audience for any alcohol-related content. Where the platform's emotional
          scenarios overlap with young adult life — exams, first jobs, friendships — the objects designed for those
          scenarios in the Supply line can be zero-proof, and their presentation is reviewed so that nothing about them
          markets alcohol to anyone underage.
        </p>
      </Sec>

      <Sec n={2} title={zh ? "年龄门" : "The age gate"}>
        <p>
          Alcohol-related areas of the platform — the Maison line, the Forge, the trade market, and alcohol product
          detail pages — display an age gate that asks you to confirm you are of legal drinking age in your region
          before proceeding. The confirmation is stored only in your own browser (see the Cookie Policy) and reappears
          if you clear site data. The age gate is a first filter, not the only one: age is verified again, with
          documentation where required, during the quotation and delivery process for any physical alcohol order.
        </p>
      </Sec>

      <Sec n={3} title={zh ? "账户年龄要求" : "Account age requirements"}>
        <p>
          Registration is open only to persons who have reached the age of majority and the legal drinking age
          applicable to them, whichever is higher. The platform does not knowingly collect personal data from minors.
          If we learn that an account belongs to a minor, the account is closed, its data is deleted in line with the
          Privacy Policy, and any pending reservations are released. A parent or guardian who believes a minor has
          created an account can contact us through the channel in Section 6 for immediate handling.
        </p>
      </Sec>

      <Sec n={4} title={zh ? "零酒精产品线政策" : "Zero-proof product line policy"}>
        <p>
          The emotional supply line includes zero-proof objects: sparkling rations, fragrance directions, label and
          card sets, and digital badges. For these, the platform applies three rules:
        </p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>
            zero-proof objects are clearly labeled as containing no alcohol, and their design must not imitate an
            alcoholic product's presentation in a way that could confuse;
          </li>
          <li>
            zero-proof objects aimed at scenarios common among students (such as exam season) must carry no alcohol
            imagery, cross-promotion, or upsell into alcoholic lines;
          </li>
          <li>
            purchasing a zero-proof object still requires an adult account holder; the platform's account rules do not
            relax for zero-proof products.
          </li>
        </ul>
      </Sec>

      <Sec n={5} title={zh ? "不向未成年人营销" : "No marketing to minors"}>
        <p>
          The operator does not target advertising or social content at minors, does not use imagery, characters, or
          language designed to appeal primarily to minors in alcohol-related material, and does not place alcohol
          content in channels directed at underage audiences. Community content follows the same rule: briefs,
          projects, label copy, and market listings that depict, address, or appeal to minors in connection with
          alcohol are rejected in review.
        </p>
      </Sec>

      <Sec n={6} title={zh ? "共创项目的未成年人安全评审" : "Co-creation minor-safety review"}>
        <p>
          Every co-creation project passes a dedicated <span className="text-porcelain">minor-safety review
          dimension</span> before its public page opens, separate from the sensitive-content and alcohol-compliance
          dimensions. Reviewers check the project's concept, imagery, emotional framing, and audience: a zero-proof
          exam-season project can pass with student framing precisely because it contains no alcohol; an alcoholic
          project with the same framing would be rejected or returned for revision. Review outcomes and reviewer notes
          are retained in the moderation log.
        </p>
      </Sec>

      <Sec n={7} title={zh ? "举报渠道" : "Reporting channel"}>
        <p>
          If you encounter content, a project, or an account that endangers minors — underage use, marketing that
          appeals to minors, or a minor holding an account — report it to{" "}
          <span className="text-porcelain">concierge@zotaix.example</span> with the subject line "Minor safety" and the
          page or object reference. Minor-safety reports are prioritized ahead of all other moderation queues, receive
          a reply, and, where the law requires, are escalated to the competent authorities.
        </p>
      </Sec>

      <Sec n={8} title={zh ? "联系" : "Contact"}>
        <p>
          Questions from parents, guardians, educators, or youth protection bodies are welcome at
          concierge@zotaix.example or through the channels on the Contact page. The human concierge responds on
          business days, 10:00–19:00 CST.
        </p>
      </Sec>
    </article>
  );
}
