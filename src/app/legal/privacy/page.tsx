import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Notice } from "@/components/ui";
import { profileNotice } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Privacy Policy",
  description:
    "How ZOTAIX collects, uses, protects, and deletes your data: account details, optional self-expression tags, preferences, conversations, memory controls, retention, sharing, and your rights.",
  path: "/legal/privacy",
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

export default async function PrivacyPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const doc = db().legal_docs.find((d) => d.slug === "privacy");

  return (
    <article className="space-y-10">
      <header>
        <h1 className="font-display text-3xl text-porcelain sm:text-4xl">{zh ? "隐私政策" : "Privacy Policy"}</h1>
        <p className="mt-3 text-xs uppercase tracking-[0.2em] text-gold">
          {zh ? "版本" : "Version"} {doc?.version ?? "1.0"} · {zh ? "生效日期" : "Effective date"}{" "}
          {doc?.effective_date ?? "2026-05-01"}
        </p>
        <p className="mt-4 text-sm leading-relaxed text-mist">
          {zh
            ? "ZOTAIX 的定制建立在理解之上，而理解建立在信任之上。本政策说明我们收集什么、为什么收集、你如何控制，以及我们绝不做什么。"
            : "ZOTAIX customization is built on understanding, and understanding is built on trust. This policy explains what we collect, why, how you stay in control, and what we never do."}
        </p>
      </header>

      <Sec n={1} title={zh ? "我们收集的数据" : "Data we collect"}>
        <p>We collect only what the service needs, in five groups:</p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>
            <span className="text-porcelain">Account data.</span> Email address, nickname, hashed password, account
            tier, and membership level. Guests are identified only by an anonymous visitor identifier used for the
            visitor quota.
          </li>
          <li>
            <span className="text-porcelain">Optional self-expression tags.</span> If you choose to add them: MBTI,
            zodiac, blood type, age range, preferred form of address, favorite colors, scent and drink preferences,
            music, films, cities, literary imagery, visual and gifting preferences, budget range, emotional state,
            common scenarios, and personality tags. Every one of these is optional.
          </li>
          <li>
            <span className="text-porcelain">Relationship profiles.</span> Notes you keep about people you create
            objects for — their nickname, preferences, and important dates. These are private by default.
          </li>
          <li>
            <span className="text-porcelain">Conversations and creations.</span> Your concierge conversations,
            generated proposals, saved object drafts, design versions, co-creation activity, and Reserve records.
          </li>
          <li>
            <span className="text-porcelain">Usage and technical logs.</span> Generation counts against your quota,
            token usage, timestamps, and the technical data needed to keep the service secure and functioning.
          </li>
        </ul>
        <Notice tone="gold">{zh ? profileNotice.zh : profileNotice.en}</Notice>
      </Sec>

      <Sec n={2} title={zh ? "使用目的" : "Purposes"}>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>Generating personalized proposals: the concierge reads your brief, tags, and (if enabled) memory to shape tone and content.</li>
          <li>Operating your archive: drafts, versions, Reserve records, and certificates you asked us to keep.</li>
          <li>Enforcing quotas and membership benefits.</li>
          <li>Human review of published, listed, or ordered content for compliance and safety.</li>
          <li>Fulfilling quotations and deliveries you have accepted, including age and region checks for alcohol.</li>
          <li>Security, fraud prevention, and legal compliance.</li>
        </ul>
        <p>
          We do not use your data to build advertising profiles, and self-expression tags never affect pricing or
          access — they influence tone and imagery only.
        </p>
      </Sec>

      <Sec n={3} title={zh ? "记忆控制" : "Memory controls"}>
        <p>You decide how much the concierge remembers. At any time you can:</p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li><span className="text-porcelain">Skip</span> — leave any profile field empty; nothing requires an answer.</li>
          <li><span className="text-porcelain">Use once</span> — give context inside a single conversation without saving it to your profile.</li>
          <li><span className="text-porcelain">Save</span> — store a preference so future proposals can use it.</li>
          <li><span className="text-porcelain">Delete</span> — remove any saved tag, preference, relationship profile, or conversation.</li>
          <li><span className="text-porcelain">Disable memory</span> — switch off profile memory entirely; the concierge then works only from what you say in the moment.</li>
          <li><span className="text-porcelain">Keep out of co-creation</span> — set your profile privacy so nothing from it is used in co-creation publishing.</li>
          <li><span className="text-porcelain">No public display</span> — keep drafts and Reserve records private; public visibility is always an explicit choice per object.</li>
          <li><span className="text-porcelain">Export</span> — request a copy of your profile, conversations, drafts, and records in a portable format.</li>
        </ul>
      </Sec>

      <Sec n={4} title={zh ? "保留期限" : "Retention"}>
        <p>
          Account and profile data are kept while your account exists and deleted when you close it, except where law
          requires longer retention (for example, invoicing records for accepted quotations). Conversations and drafts
          are kept until you delete them or close your account. Reserve records follow the Reserve Archive Rules:
          deleting a record removes its content, while its serial number is retired rather than reissued, so no other
          object can ever claim your identifier. Anonymous usage statistics that no longer identify you may be kept for
          service planning.
        </p>
      </Sec>

      <Sec n={5} title={zh ? "共享" : "Sharing"}>
        <p>
          <span className="text-porcelain">Your data is never sold.</span> We share it only with:
        </p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>
            <span className="text-porcelain">AI providers, when configured.</span> Where the operator has connected an
            external AI provider, the text needed to answer your brief is processed by that provider under a data
            processing agreement. Where no external provider is configured, generation runs on the platform's own
            deterministic atelier engine and your briefs stay within the platform.
          </li>
          <li>
            <span className="text-porcelain">Payment processors, when configured.</span> Where a payment channel is
            connected, the processor receives the data required to complete your payment. Where none is connected,
            orders are confirmed by a human concierge and no payment data flows through the platform.
          </li>
          <li>
            <span className="text-porcelain">Logistics and production partners</span> — only for orders you have
            accepted, and only the delivery data they need.
          </li>
          <li>
            <span className="text-porcelain">Authorities</span> — where a valid legal obligation requires it.
          </li>
        </ul>
        <p>There are no third-party advertising or tracking partners; see the Cookie Policy.</p>
      </Sec>

      <Sec n={6} title={zh ? "你的权利" : "Your rights"}>
        <p>
          You can access, correct, export, and delete your data. Access and export cover your profile, conversations,
          drafts, design versions, and Reserve records. Deletion requests are honored as described in Section 4.
          Exercise these rights directly in your profile settings or by writing to concierge@zotaix.example; we respond
          within the period required by applicable law, and in any case aim for fourteen days. You may also lodge a
          complaint with the supervisory authority competent at the operator's registered domicile or at your place of
          residence.
        </p>
      </Sec>

      <Sec n={7} title={zh ? "未成年人" : "Children"}>
        <p>
          ZOTAIX does not offer accounts to minors and does not knowingly collect their data. The platform operates an
          age gate on alcohol-related areas, and any account found to belong to a minor is closed and its data deleted.
          Details are in the Minor Protection Notice.
        </p>
      </Sec>

      <Sec n={8} title={zh ? "安全" : "Security"}>
        <p>
          Passwords are stored only as salted hashes. Access to personal data inside the operating team is restricted
          to roles that need it, private records are never exposed on public pages, and transport is encrypted. No
          system is perfectly secure; if a breach affecting your data occurs, we will notify you and the competent
          authority as required by law.
        </p>
      </Sec>

      <Sec n={9} title={zh ? "变更与联系" : "Changes and contact"}>
        <p>
          The version and effective date above identify the current policy; material changes are announced before they
          take effect. Privacy questions and requests: concierge@zotaix.example, or the channels on the Contact page.
        </p>
      </Sec>
    </article>
  );
}
