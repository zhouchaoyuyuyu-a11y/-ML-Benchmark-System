import Link from "next/link";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import AgeGate from "@/components/AgeGate";
import QRCodeBox from "@/components/QRCodeBox";
import { Card, DefinitionRow, Notice, ProgressBar, Section, StatusPill, Tag } from "@/components/ui";
import MarketActions from "./MarketActions";
import { complianceNotice, pick } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db, type Database } from "@/lib/store";
import type { CoCreationProject, ObjectDraft, ReserveRecord } from "@/lib/types";

type MarketItem =
  | { kind: "draft"; draft: ObjectDraft }
  | { kind: "project"; project: CoCreationProject }
  | { kind: "record"; record: ReserveRecord };

function findMarketItem(data: Database, id: string): MarketItem | null {
  if (id.startsWith("dft")) {
    const draft = data.object_drafts.find((d) => d.id === id && d.public_visible);
    return draft ? { kind: "draft", draft } : null;
  }
  if (id.startsWith("ccp")) {
    const project = data.co_creation_projects.find(
      (p) => p.id === id && p.public_visible && p.review_status === "approved"
    );
    return project ? { kind: "project", project } : null;
  }
  if (id.startsWith("rsv")) {
    const record = data.reserve_records.find((r) => r.id === id && r.privacy_level === "public");
    return record ? { kind: "record", record } : null;
  }
  return null;
}

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const { id } = await params;
  const item = findMarketItem(db(), id);
  if (!item) {
    return pageMetadata({
      title: "Creative Market listing",
      description: "Public creative proposals, founder rights, and open Reserve archives on the ZOTAIX Creative Market.",
      path: `/market/${id}`,
    });
  }
  const title =
    item.kind === "draft"
      ? `${item.draft.title} — creative proposal`
      : item.kind === "project"
        ? `${item.project.title} — founder rights & participation`
        : `${item.record.object_name} — public archive`;
  const description =
    item.kind === "draft"
      ? item.draft.label_copy ?? item.draft.scene ?? "A public creative proposal on the ZOTAIX Creative Market."
      : item.kind === "project"
        ? item.project.concept.slice(0, 160)
        : item.record.product_direction ??
          item.record.label_copy ??
          "A public Reserve archive page on the ZOTAIX Creative Market.";
  return pageMetadata({ title, description, path: `/market/${id}` });
}

export default async function MarketDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const item = findMarketItem(data, id);
  if (!item) notFound();

  const badge =
    item.kind === "draft"
      ? { text: zh ? "创意提案" : "Creative proposal", tone: "default" as const }
      : item.kind === "project"
        ? { text: zh ? "发起人权益 / 参与" : "Founder rights / participation", tone: "gold" as const }
        : { text: zh ? "公开档案" : "Public archive", tone: "jade" as const };

  const title = item.kind === "draft" ? item.draft.title : item.kind === "project" ? item.project.title : item.record.object_name;

  // Human review status: projects carry it directly; drafts and records surface
  // the moderation log entry created when the item was cleared for public display.
  const modLog = data.moderation_logs.find((m) => m.object_id === id);
  const reviewStatus = item.kind === "project" ? item.project.review_status : modLog?.review_status ?? "approved";

  const tags = item.kind === "draft" ? item.draft.emotion_tags : item.kind === "project" ? item.project.emotion_tags : item.record.emotion_tags;

  return (
    <>
      <AgeGate zh={zh} />

      <div className="zx-grid-bg border-b border-hairline">
        <Section className="py-12 sm:py-16">
          <Link href="/market" className="text-xs text-mist transition-colors hover:text-gold">
            ← {zh ? "返回创意市场" : "Back to Creative Market"}
          </Link>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <Tag tone={badge.tone}>{badge.text}</Tag>
            <span className="text-xs text-mist">{id}</span>
          </div>
          <h1 className="font-display mt-4 max-w-4xl text-3xl leading-tight text-porcelain sm:text-4xl">{title}</h1>
          {tags.length > 0 && (
            <div className="mt-4 flex flex-wrap gap-2">
              {tags.map((t) => (
                <Tag key={t}>{t}</Tag>
              ))}
            </div>
          )}
        </Section>
      </div>

      <Section className="py-10 sm:py-14">
        <div className="grid gap-8 lg:grid-cols-[1fr_340px]">
          {/* Details */}
          <div className="space-y-6">
            {item.kind === "draft" && (
              <Card>
                <p className="font-display text-lg text-porcelain">{zh ? "提案详情" : "Proposal details"}</p>
                {item.draft.label_copy && (
                  <blockquote className="mt-4 border-l-2 border-gold/50 pl-4 text-base italic leading-relaxed text-porcelain">
                    “{item.draft.label_copy}”
                  </blockquote>
                )}
                <dl className="mt-4">
                  <DefinitionRow label={zh ? "对象类型" : "Object type"}>{item.draft.object_type.replace(/_/g, " ")}</DefinitionRow>
                  {item.draft.scene && <DefinitionRow label={zh ? "场景" : "Scene"}>{item.draft.scene}</DefinitionRow>}
                  {item.draft.recipient && <DefinitionRow label={zh ? "赠予对象" : "Recipient"}>{item.draft.recipient}</DefinitionRow>}
                  {item.draft.budget && <DefinitionRow label={zh ? "预算" : "Budget"}>{item.draft.budget}</DefinitionRow>}
                  {item.draft.liquid_direction && (
                    <DefinitionRow label={zh ? "液体方向" : "Liquid direction"}>{item.draft.liquid_direction}</DefinitionRow>
                  )}
                  {item.draft.scent_direction && (
                    <DefinitionRow label={zh ? "香氛方向" : "Scent direction"}>{item.draft.scent_direction}</DefinitionRow>
                  )}
                  {item.draft.visual_style && (
                    <DefinitionRow label={zh ? "视觉风格" : "Visual style"}>{item.draft.visual_style}</DefinitionRow>
                  )}
                  {item.draft.names && item.draft.names.length > 0 && (
                    <DefinitionRow label={zh ? "候选命名" : "Name candidates"}>{item.draft.names.join(" · ")}</DefinitionRow>
                  )}
                  <DefinitionRow label={zh ? "草稿状态" : "Draft status"}>
                    <StatusPill status={item.draft.status} />
                  </DefinitionRow>
                  <DefinitionRow label={zh ? "创建于" : "Created"}>{item.draft.created_at.slice(0, 10)}</DefinitionRow>
                </dl>
              </Card>
            )}

            {item.kind === "project" && (
              <Card>
                <p className="font-display text-lg text-porcelain">{zh ? "共创项目详情" : "Co-creation project details"}</p>
                <p className="mt-3 text-sm leading-relaxed text-mist">{item.project.concept}</p>
                <div className="mt-5 space-y-1.5">
                  <div className="flex justify-between text-xs text-mist">
                    <span>
                      {item.project.current_quantity}/{item.project.target_quantity} {zh ? "已预订" : "reserved"}
                    </span>
                    <span>
                      {item.project.supporters} {zh ? "位支持者" : "supporters"} · {item.project.votes} {zh ? "票" : "votes"}
                    </span>
                  </div>
                  <ProgressBar value={item.project.current_quantity} max={item.project.target_quantity} />
                </div>
                <dl className="mt-4">
                  <DefinitionRow label={zh ? "产品类型" : "Product type"}>{item.project.product_type}</DefinitionRow>
                  <DefinitionRow label={zh ? "发起人权益" : "Founder benefit"}>{item.project.founder_benefit}</DefinitionRow>
                  <DefinitionRow label={zh ? "项目状态" : "Project status"}>
                    <StatusPill status={item.project.status} />
                  </DefinitionRow>
                  <DefinitionRow label={zh ? "创建于" : "Created"}>{item.project.created_at.slice(0, 10)}</DefinitionRow>
                </dl>
              </Card>
            )}

            {item.kind === "record" && (
              <Card>
                <p className="font-display text-lg text-porcelain">{zh ? "档案详情" : "Archive details"}</p>
                {item.record.label_copy && (
                  <blockquote className="mt-4 border-l-2 border-gold/50 pl-4 text-base italic leading-relaxed text-porcelain">
                    “{item.record.label_copy}”
                  </blockquote>
                )}
                <dl className="mt-4">
                  <DefinitionRow label="ZOTAIX ID">{item.record.zotaix_id}</DefinitionRow>
                  <DefinitionRow label={zh ? "对象类型" : "Object type"}>{item.record.object_type.replace(/_/g, " ")}</DefinitionRow>
                  {item.record.relationship_scene && (
                    <DefinitionRow label={zh ? "关系场景" : "Relationship scene"}>{item.record.relationship_scene}</DefinitionRow>
                  )}
                  {item.record.product_direction && (
                    <DefinitionRow label={zh ? "产品方向" : "Product direction"}>{item.record.product_direction}</DefinitionRow>
                  )}
                  {item.record.liquid_direction && (
                    <DefinitionRow label={zh ? "液体方向" : "Liquid direction"}>{item.record.liquid_direction}</DefinitionRow>
                  )}
                  {item.record.scent_direction && (
                    <DefinitionRow label={zh ? "香氛方向" : "Scent direction"}>{item.record.scent_direction}</DefinitionRow>
                  )}
                  {item.record.visual_style && (
                    <DefinitionRow label={zh ? "视觉风格" : "Visual style"}>{item.record.visual_style}</DefinitionRow>
                  )}
                  {item.record.batch_id && <DefinitionRow label={zh ? "批次" : "Batch"}>{item.record.batch_id}</DefinitionRow>}
                  <DefinitionRow label={zh ? "交付状态" : "Delivery status"}>
                    <StatusPill status={item.record.delivery_status} />
                  </DefinitionRow>
                  <DefinitionRow label={zh ? "售后状态" : "Aftercare"}>
                    <StatusPill status={item.record.aftercare_status} />
                  </DefinitionRow>
                  <DefinitionRow label={zh ? "可补铸" : "Replenishment"}>
                    {item.record.repurchase_eligible
                      ? zh
                        ? "可通过 Trade 补铸"
                        : "Eligible via Trade"
                      : zh
                        ? "此对象为纯数字档案"
                        : "This object lives as a digital archive"}
                  </DefinitionRow>
                  <DefinitionRow label={zh ? "归档于" : "Archived"}>{item.record.created_at.slice(0, 10)}</DefinitionRow>
                </dl>
              </Card>
            )}

            {/* Rights explanation */}
            <Card>
              <p className="font-display text-base text-porcelain">{zh ? "权益说明" : "How rights work here"}</p>
              <ul className="mt-3 space-y-2 text-sm leading-relaxed text-mist">
                <li className="flex items-start gap-2.5">
                  <span className="mt-0.5 text-gold">❖</span>
                  <span>
                    {zh
                      ? "授权 ≠ 所有权转移：授权是限定范围的商用使用权，作品与档案身份始终属于创作者。"
                      : "Authorization ≠ ownership transfer: an authorization is a scoped commercial-use right; the work and its Reserve identity stay with the creator."}
                  </span>
                </li>
                <li className="flex items-start gap-2.5">
                  <span className="mt-0.5 text-gold">◆</span>
                  <span>
                    {zh
                      ? "所有授权与实体执行必须经过平台人工审核与合规审查。"
                      : "Every authorization and physical execution passes platform human review and compliance checks."}
                  </span>
                </li>
                <li className="flex items-start gap-2.5">
                  <span className="mt-0.5 text-gold">▣</span>
                  <span>
                    {zh
                      ? "设计师收益分成：授权商用经审核通过后，创作者按协议约定获得收益分成，比例逐案确定。"
                      : "Designer income share: after an authorization passes human review, the creator receives an agreed share of commercial proceeds — configurable per agreement."}
                  </span>
                </li>
              </ul>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            <Card>
              <p className="text-xs uppercase tracking-wider text-mist">{zh ? "人工审核状态" : "Human review status"}</p>
              <div className="mt-2">
                <StatusPill status={reviewStatus} />
              </div>
              <p className="mt-3 text-xs leading-relaxed text-mist">
                {zh
                  ? "公开陈列的对象已通过公开展示审核；任何商用授权或实体执行都会触发新一轮人工审核。"
                  : "Publicly displayed items have passed display review; any commercial authorization or physical execution triggers a fresh round of human review."}
              </p>
              {modLog?.reviewer_note && <p className="mt-2 text-xs italic leading-relaxed text-mist">“{modLog.reviewer_note}”</p>}
            </Card>

            {item.kind === "record" && (
              <Card className="flex flex-col items-center text-center">
                <QRCodeBox seed={item.record.qr_nfc_id} label={item.record.qr_nfc_id} size={180} />
                <p className="mt-3 text-xs leading-relaxed text-mist">
                  {zh
                    ? "扫描实体上的 QR / NFC 标识即可打开这一页——档案与对象终身绑定。"
                    : "Scanning the QR / NFC mark on the physical object opens this page — the archive stays bound to the object for life."}
                </p>
              </Card>
            )}

            <MarketActions
              zh={zh}
              itemId={id}
              itemTitle={title}
              draftId={item.kind === "draft" ? item.draft.id : item.kind === "record" ? item.record.object_draft_id : undefined}
              projectId={item.kind === "project" ? item.project.id : undefined}
            />
          </div>
        </div>
      </Section>

      <Section className="pb-14 sm:pb-20">
        <div className="space-y-4">
          <Notice tone="ember" title={zh ? "风险提示" : "Risk notice"}>
            {zh
              ? "本页不是酒类销售页：创意市场不进行任何酒类转售，所有实体执行都必须经过平台人工审核与年龄、地区合规审查。"
              : "This is not an alcohol sales page: the Creative Market performs no alcohol resale, and all physical execution passes platform human review plus age and region compliance checks."}
          </Notice>
          <Notice tone="gold" title={zh ? "合规声明" : "Compliance"}>
            {pick(locale, complianceNotice)}
          </Notice>
        </div>
      </Section>
    </>
  );
}
