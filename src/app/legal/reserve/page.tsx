import type { Metadata } from "next";
import type { ReactNode } from "react";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Reserve Archive Rules",
  description:
    "The rules of the Reserve archive: what a record contains, ZOTAIX IDs and QR/NFC binding, public versus private records, certificates, aftercare, portability, and deletion.",
  path: "/legal/reserve",
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

export default async function ReserveRulesPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const doc = db().legal_docs.find((d) => d.slug === "reserve");

  return (
    <article className="space-y-10">
      <header>
        <h1 className="font-display text-3xl text-porcelain sm:text-4xl">
          {zh ? "档案规则" : "Reserve Archive Rules"}
        </h1>
        <p className="mt-3 text-xs uppercase tracking-[0.2em] text-gold">
          {zh ? "版本" : "Version"} {doc?.version ?? "1.0"} · {zh ? "生效日期" : "Effective date"}{" "}
          {doc?.effective_date ?? "2026-05-01"}
        </p>
        <p className="mt-4 text-sm leading-relaxed text-mist">
          {zh
            ? "档案馆（Reserve）为每一个被保存的对象建立身份。本规则说明档案的内容、编号、公开与私密、证书、售后与删除。"
            : "Reserve gives every kept object an identity. These rules define what a record contains, how it is numbered, public versus private status, certificates, aftercare, and deletion."}
        </p>
      </header>

      <Sec n={1} title={zh ? "档案记录包含什么" : "What a record contains"}>
        <p>A Reserve record is the durable identity of one object. Depending on the object type, it holds:</p>
        <ul className="list-disc space-y-1.5 pl-5">
          <li>the object's name, type (spirit, fragrance, bottle, gift box, label, badge, co-creation unit, or design version), and emotion tags;</li>
          <li>the relationship scene it was made for — who, and which moment;</li>
          <li>the product direction: liquid direction, scent direction, visual style, and label copy as archived;</li>
          <li>the linked draft and design version, so the winning version is fixed in time;</li>
          <li>the batch identifier for produced objects;</li>
          <li>the ZOTAIX ID, QR/NFC binding, and certificate reference;</li>
          <li>status fields: privacy level, delivery status, co-creation eligibility, replenishment eligibility, and aftercare status.</li>
        </ul>
        <p>
          A record can exist for a purely digital object — an archived direction, a badge — as well as for a produced
          one. The record, not the bottle, is the durable artifact.
        </p>
      </Sec>

      <Sec n={2} title={zh ? "ZOTAIX ID 与 QR/NFC 绑定" : "ZOTAIX ID and QR/NFC binding"}>
        <p>
          Every record receives a <span className="text-porcelain">ZOTAIX ID</span> — a serial of the form
          ZX-YYYY-MMDD-NNNN — issued once and never reissued. Produced objects additionally carry a QR code (and, where
          the packaging supports it, an NFC tag) bound to the record: scanning it opens the record's page. The binding
          is one-to-one; a QR/NFC identifier belongs to exactly one record for the life of the platform. If a physical
          tag is damaged, the concierge can verify ownership and issue a replacement tag bound to the same record — the
          ID itself never changes.
        </p>
      </Sec>

      <Sec n={3} title={zh ? "公开与私密" : "Public versus private records"}>
        <p>
          Every record is <span className="text-porcelain">private by default</span>. A private record is visible only
          to its owner and, where scanning a physical object's code, shows only a neutral validity confirmation — the
          serial exists — with none of the record's content. Setting a record public is an explicit choice by the
          owner: the record's page then becomes shareable and scannable in full, and may appear in public archive
          listings. Public status passes a public-display review before the page opens, and the owner can return a
          record to private at any time, which withdraws the public page.
        </p>
      </Sec>

      <Sec n={4} title={zh ? "证书" : "Certificates"}>
        <p>
          Records of eligible plans and orders carry a certificate: a page (and printable rendering) stating the
          object's name, ZOTAIX ID, archived directions, label copy, and issue date. Certificates attest that the
          record was archived on the platform at the stated time; they are not appraisals, valuations, or guarantees of
          market worth. A certificate is regenerated if the record's owner corrects archived content, and the
          certificate page always reflects the current record.
        </p>
      </Sec>

      <Sec n={5} title={zh ? "售后与补铸" : "Aftercare and replenishment"}>
        <p>
          Aftercare attaches to the record, not the object. Where an order's quotation includes aftercare, the record
          shows aftercare status active, and its owner can raise care requests (packaging repair, tag replacement,
          delivery issues) through the concierge for the stated period. Records marked{" "}
          <span className="text-porcelain">replenishment eligible</span> carry a replenishment entry: the owner can
          request the same object again — same base, same expression, new pour — and the request enters the standard
          quotation path, including all alcohol compliance checks where applicable. Replenishment recreates the
          expression faithfully; batch-level variation in the base liquid is stated in the quotation.
        </p>
      </Sec>

      <Sec n={6} title={zh ? "数据可携带" : "Data portability"}>
        <p>
          You can export your Reserve records — content, directions, label copy, serials, and certificate data — in a
          portable format at any time from your profile or by request to the concierge, as part of the export right in
          the Privacy Policy. Exported copies are yours to keep; the platform's record remains the authoritative
          version for QR/NFC verification.
        </p>
      </Sec>

      <Sec n={7} title={zh ? "删除：内容移除与编号封存" : "Deletion: record removal versus serial retirement"}>
        <p>
          Deleting a record has two distinct effects. <span className="text-porcelain">Record removal</span> deletes
          the record's content — directions, copy, scene, tags, certificate data — from the archive and withdraws any
          public page. <span className="text-porcelain">Serial retirement</span> then permanently retires the ZOTAIX ID
          and its QR/NFC binding: the identifier is never reassigned, and a scan of a retired serial shows only that
          the serial existed and was retired, with no content. This split protects both your privacy (the content is
          gone) and the integrity of every other serial (no identifier can ever point to a different object than the
          one it was born with). Serial retirement is irreversible; the concierge confirms it with you before
          executing.
        </p>
      </Sec>

      <Sec n={8} title={zh ? "准确性与联系" : "Accuracy and contact"}>
        <p>
          Owners may correct archived content that was recorded in error; corrections are noted on the record.
          Questions about records, certificates, bindings, or deletion: concierge@zotaix.example, or the channels on
          the Contact page.
        </p>
      </Sec>
    </article>
  );
}
