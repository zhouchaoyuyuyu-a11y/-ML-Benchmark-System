import Link from "next/link";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import QRCodeBox from "@/components/QRCodeBox";
import { Card, DefinitionRow, Meridian, Notice, Section, StatusPill, Tag } from "@/components/ui";
import ShareRow from "./ShareRow";
import { getSessionUser } from "@/lib/auth";
import { siteUrl } from "@/lib/config";
import { complianceNotice, pick } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const { id } = await params;
  const record = db().reserve_records.find((r) => r.id === id);
  if (!record || record.privacy_level !== "public") {
    return pageMetadata({
      title: "Reserve certificate",
      description: "A ZOTAIX Reserve certificate: the lifetime identity page of a personalized object.",
      path: `/reserve/${id}`,
    });
  }
  return pageMetadata({
    title: `${record.object_name} — Reserve certificate ${record.zotaix_id}`,
    description:
      record.label_copy ??
      record.product_direction ??
      "A public ZOTAIX Reserve certificate: emotional origin, product direction, QR identity, and replenishment entry.",
    path: `/reserve/${id}`,
    image: `${siteUrl}/api/og?title=${encodeURIComponent(record.object_name)}&subtitle=${encodeURIComponent(record.zotaix_id)}`,
  });
}

export default async function ReserveCertificatePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const record = data.reserve_records.find((r) => r.id === id);
  if (!record) notFound();

  const user = await getSessionUser();
  const isOwner = !!user && user.id === record.user_id;
  if (record.privacy_level === "private" && !isOwner) notFound();

  const version = record.design_version_id
    ? data.design_versions.find((v) => v.id === record.design_version_id)
    : undefined;

  const cardUrl = `/api/card?${new URLSearchParams({
    copy: record.label_copy ?? record.object_name,
    mark: record.zotaix_id,
    keywords: record.emotion_tags.join(" · "),
  }).toString()}`;

  return (
    <>
      <div className="zx-grid-bg border-b border-hairline">
        <Section className="py-10 sm:py-14">
          <Link href="/reserve" className="text-xs text-mist transition-colors hover:text-gold">
            ← {zh ? "返回档案馆" : "Back to the Reserve"}
          </Link>
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <Tag tone="gold">{zh ? "Reserve 证书" : "Reserve certificate"}</Tag>
            <Tag tone={record.privacy_level === "public" ? "jade" : "default"}>
              {record.privacy_level === "public" ? (zh ? "公开档案" : "public record") : zh ? "私密档案" : "private record"}
            </Tag>
            <StatusPill status={record.delivery_status} />
          </div>
        </Section>
      </div>

      <Section className="py-10 sm:py-16">
        {/* Certificate body */}
        <div className="zx-card mx-auto max-w-3xl overflow-hidden">
          <div className="border-b border-hairline px-6 py-8 text-center sm:px-10 sm:py-10">
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-gold">
              {zh ? "ZOTAIX 档案证书" : "ZOTAIX Reserve Certificate"}
            </p>
            <p className="font-display mt-4 text-3xl tracking-[0.08em] text-porcelain sm:text-4xl">{record.zotaix_id}</p>
            <Meridian className="mx-auto my-6 max-w-[200px]" />
            <h1 className="font-display text-2xl leading-snug text-porcelain sm:text-3xl">{record.object_name}</h1>
            {record.relationship_scene && (
              <p className="mt-2 text-sm text-mist">{record.relationship_scene}</p>
            )}
            <div className="mt-6 flex justify-center">
              <QRCodeBox seed={record.qr_nfc_id} label={record.qr_nfc_id} size={180} />
            </div>
            <p className="mx-auto mt-4 max-w-md text-xs leading-relaxed text-mist">
              {zh
                ? "扫描实体对象上的 QR / NFC 标识即打开本页——档案与对象终身绑定。"
                : "Scanning the QR / NFC mark on the physical object opens this page — the archive stays bound to the object for life."}
            </p>
          </div>

          <div className="px-6 py-6 sm:px-10 sm:py-8">
            {record.label_copy && (
              <blockquote className="border-l-2 border-gold/50 pl-4 text-base italic leading-relaxed text-porcelain sm:text-lg">
                “{record.label_copy}”
              </blockquote>
            )}
            <dl className="mt-5">
              <DefinitionRow label={zh ? "对象类型" : "Object type"}>{record.object_type.replace(/_/g, " ")}</DefinitionRow>
              {record.emotion_tags.length > 0 && (
                <DefinitionRow label={zh ? "情绪关键词" : "Emotional keywords"}>
                  <span className="flex flex-wrap gap-2">
                    {record.emotion_tags.map((t) => (
                      <Tag key={t}>{t}</Tag>
                    ))}
                  </span>
                </DefinitionRow>
              )}
              {record.relationship_scene && (
                <DefinitionRow label={zh ? "关系场景" : "Relationship scene"}>{record.relationship_scene}</DefinitionRow>
              )}
              {record.liquid_direction && (
                <DefinitionRow label={zh ? "酒体方向" : "Liquid direction"}>{record.liquid_direction}</DefinitionRow>
              )}
              {record.scent_direction && (
                <DefinitionRow label={zh ? "香氛方向" : "Fragrance direction"}>{record.scent_direction}</DefinitionRow>
              )}
              {record.product_direction && (
                <DefinitionRow label={zh ? "产品方向" : "Product direction"}>{record.product_direction}</DefinitionRow>
              )}
              {record.visual_style && (
                <DefinitionRow label={zh ? "视觉风格" : "Visual style"}>{record.visual_style}</DefinitionRow>
              )}
              {version && (
                <DefinitionRow label={zh ? "设计版本" : "Design version"}>
                  {version.version_name} · <span className="text-mist">{version.version_hash}</span>
                </DefinitionRow>
              )}
              {record.batch_id && <DefinitionRow label={zh ? "生产批次" : "Batch"}>{record.batch_id}</DefinitionRow>}
              <DefinitionRow label={zh ? "交付状态" : "Delivery status"}>
                <StatusPill status={record.delivery_status} />
              </DefinitionRow>
              <DefinitionRow label={zh ? "售后状态" : "Aftercare"}>
                <StatusPill status={record.aftercare_status} />
              </DefinitionRow>
              <DefinitionRow label={zh ? "归档日期" : "Archived on"}>{record.created_at.slice(0, 10)}</DefinitionRow>
            </dl>
          </div>
        </div>

        {/* Share & actions */}
        <div className="mx-auto mt-8 max-w-3xl">
          <ShareRow
            zh={zh}
            zotaixId={record.zotaix_id}
            cardUrl={cardUrl}
            repurchaseEligible={record.repurchase_eligible}
            isPublic={record.privacy_level === "public"}
          />
        </div>

        <div className="mx-auto mt-8 max-w-3xl space-y-4">
          {record.privacy_level === "private" && (
            <Card className="!p-4">
              <p className="text-sm text-porcelain">{zh ? "私密档案" : "Private record"}</p>
              <p className="mt-1 text-xs leading-relaxed text-mist">
                {zh
                  ? "这条记录已封存为私密，仅你本人登录后可见。分享链接对其他访客不可用。"
                  : "This record is sealed as private and visible only to you while signed in. The share link stays closed to other visitors."}
              </p>
            </Card>
          )}
          {!record.repurchase_eligible && (
            <Card className="!p-4">
              <p className="text-sm text-porcelain">{zh ? "纯数字档案" : "Digital-only archive"}</p>
              <p className="mt-1 text-xs leading-relaxed text-mist">
                {zh
                  ? "这个对象以数字形态存续，不进入补铸流程。它的证书、印记与情绪卡片依然可以随时使用。"
                  : "This object lives digitally and does not enter the replenishment flow. Its certificate, digital mark, and emotional card remain available at any time."}
              </p>
            </Card>
          )}
          <Notice tone="gold" title={zh ? "合规声明" : "Compliance"}>
            {pick(locale, complianceNotice)}{" "}
            {zh
              ? "补铸涉及酒类实体交付时，适用年龄与地区合规审核。"
              : "Age and region compliance checks apply whenever a replenishment involves physical alcohol delivery."}
          </Notice>
        </div>
      </Section>
    </>
  );
}
