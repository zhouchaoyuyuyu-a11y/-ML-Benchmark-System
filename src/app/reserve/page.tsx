import Link from "next/link";
import type { Metadata } from "next";
import QRCodeBox from "@/components/QRCodeBox";
import {
  ButtonLink,
  Card,
  EmptyState,
  Meridian,
  Notice,
  PageHero,
  Section,
  SectionHeader,
  Stat,
  StatusPill,
  Tag,
} from "@/components/ui";
import ArchiveActions from "./ArchiveActions";
import { getSessionUser } from "@/lib/auth";
import { complianceNotice, pick } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Reserve — the archive where every object keeps its identity",
  description:
    "Every generated ZOTAIX object can become a Reserve record: a ZOTAIX ID, QR/NFC binding, certificate page, aftercare timeline, and a replenishment entry that turns one bottle into a ritual.",
  path: "/reserve",
  keywords: ["Reserve", "ZOTAIX ID", "certificate", "QR NFC", "aftercare", "replenishment", "archive"],
});

const RECORD_FIELDS: { en: string; zh: string; hintEn: string; hintZh: string }[] = [
  { en: "ZOTAIX ID", zh: "ZOTAIX 编号", hintEn: "Lifetime serial, e.g. ZX-2026-0611-0001", hintZh: "终身序列号，如 ZX-2026-0611-0001" },
  { en: "Object name", zh: "对象命名", hintEn: "The name you kept", hintZh: "你最终保留的名字" },
  { en: "Object type", zh: "对象类型", hintEn: "Spirit, fragrance, bottle, gift box, label, badge…", hintZh: "酒饮、香氛、瓶身、礼盒、标签、徽章…" },
  { en: "Emotional keywords", zh: "情绪关键词", hintEn: "The feelings the object was cast from", hintZh: "铸造这个对象的情绪来源" },
  { en: "Relationship scene", zh: "关系场景", hintEn: "Who it is for, and which moment", hintZh: "赠予对象与所属时刻" },
  { en: "Product direction", zh: "产品方向", hintEn: "The confirmed overall direction", hintZh: "确认后的整体方向" },
  { en: "Liquid direction", zh: "酒体方向", hintEn: "Base liquid, notes, ABV framing", hintZh: "基酒、风味与度数框架" },
  { en: "Fragrance direction", zh: "香氛方向", hintEn: "Top, heart, and base composition", hintZh: "前中后调的香氛结构" },
  { en: "Label copy", zh: "瓶身文案", hintEn: "The one line that sounds like you", hintZh: "那句像你说出来的话" },
  { en: "Visual style", zh: "视觉风格", hintEn: "Palette, typography, bottle language", hintZh: "配色、字体与瓶身语言" },
  { en: "Design version", zh: "设计版本", hintEn: "Which version won, with its hash", hintZh: "胜出的版本与版本指纹" },
  { en: "Source draft", zh: "来源草稿", hintEn: "The draft the record was archived from", hintZh: "入档前的原始草稿" },
  { en: "Batch number", zh: "生产批次", hintEn: "Assigned when a run is scheduled", hintZh: "排产后写入的批次号" },
  { en: "QR / NFC identity", zh: "QR / NFC 标识", hintEn: "Scan the object, open its certificate", hintZh: "扫描实体即打开证书页" },
  { en: "Certificate page", zh: "证书页面", hintEn: "A shareable page for public records", hintZh: "公开档案的可分享页面" },
  { en: "Privacy level", zh: "隐私等级", hintEn: "Public and shareable, or sealed private", hintZh: "公开可分享，或私密封存" },
  { en: "Co-creation eligibility", zh: "共创资格", hintEn: "Whether it can enter the pool", hintZh: "是否可进入共创池" },
  { en: "Delivery status", zh: "交付状态", hintEn: "Digital, in review, in production, delivered", hintZh: "数字、审核中、生产中、已交付" },
  { en: "Replenishment eligibility", zh: "补铸资格", hintEn: "Whether the same object can be poured again", hintZh: "同一对象能否再次铸造" },
  { en: "Aftercare status", zh: "售后状态", hintEn: "Active, expired, or digital-only", hintZh: "生效中、已到期或纯数字" },
  { en: "Archive timeline", zh: "档案时间线", hintEn: "Created and updated timestamps", hintZh: "创建与更新时间" },
];

export default async function ReservePage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const user = await getSessionUser();

  const records = user
    ? data.reserve_records.filter((r) => r.user_id === user.id)
    : data.reserve_records.filter((r) => r.privacy_level === "public");

  const unarchivedDrafts = user
    ? data.object_drafts.filter(
        (d) => d.user_id === user.id && !data.reserve_records.some((r) => r.object_draft_id === d.id)
      )
    : [];

  const inProduction = records.filter((r) => r.delivery_status === "in_production" || r.delivery_status === "pending_review").length;
  const aftercareActive = records.filter((r) => r.aftercare_status === "active").length;
  const replenishable = records.filter((r) => r.repurchase_eligible).length;

  return (
    <>
      <PageHero
        eyebrow={zh ? "Reserve · 档案馆" : "Reserve · the archive"}
        title={
          zh
            ? "每一个被创造的对象，都值得一个终身身份"
            : "Every object you create deserves an identity that outlasts it"
        }
        description={
          zh
            ? "在 ZOTAIX，任何生成的对象都可以入档为一条 Reserve 记录：ZOTAIX 编号、QR/NFC 绑定、证书页面、售后时间线、补铸入口与会员权益。瓶会空，香会散——记录它为何存在的档案，才是留下来的部分。"
            : "On ZOTAIX, every generated object can become a Reserve record: a ZOTAIX ID, a QR/NFC binding, a certificate page, an aftercare timeline, a replenishment entry, and membership benefits. A bottle empties and a fragrance fades — the record of why they existed is the part that lasts."
        }
      >
        <ButtonLink href="/concierge" variant="gold">
          {zh ? "创造一个对象" : "Create an object"}
        </ButtonLink>
        <ButtonLink href="#records" variant="outline">
          {zh ? "查看档案记录" : "Browse records"}
        </ButtonLink>
        <ButtonLink href="/membership" variant="outline">
          {zh ? "核心序列权益" : "Core Sequence benefits"}
        </ButtonLink>
      </PageHero>

      {/* Records */}
      <Section id="records" className="scroll-mt-24 py-14 sm:py-20">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <SectionHeader
            eyebrow={user ? (zh ? "你的档案" : "Your archive") : zh ? "公开档案" : "Public archive"}
            title={
              user
                ? zh
                  ? `${user.nickname} 的 Reserve 记录`
                  : `Reserve records of ${user.nickname}`
                : zh
                  ? "被选择保存的公开时刻"
                  : "Public moments someone chose to keep"
            }
            description={
              user
                ? zh
                  ? "每条记录都是一个终身档案页：可扫码、可分享（公开档案）、可补铸。"
                  : "Each record is a lifetime archive page — scannable, shareable when public, and replenishable."
                : zh
                  ? "公开档案页可被任何人扫码访问。注册后，你自己的对象也会在这里获得身份。"
                  : "Public certificate pages open from any QR scan. Register and your own objects earn their identity here too."
            }
          />
          {!user && (
            <ButtonLink href="/register" variant="gold">
              {zh ? "注册并开始入档" : "Register to start archiving"}
            </ButtonLink>
          )}
        </div>

        <div className="mt-8 grid gap-4 sm:grid-cols-3">
          <Stat label={zh ? "档案总数" : "Records"} value={String(records.length)} hint={zh ? "已入档对象" : "archived objects"} />
          <Stat
            label={zh ? "生产 / 审核中" : "In production / review"}
            value={String(inProduction)}
            hint={zh ? "正在成为实体" : "becoming physical"}
          />
          <Stat
            label={zh ? "可补铸" : "Replenishable"}
            value={String(replenishable)}
            hint={zh ? `售后生效中 ${aftercareActive}` : `${aftercareActive} with active aftercare`}
          />
        </div>

        {records.length > 0 ? (
          <div className="mt-8 grid gap-4 lg:grid-cols-2">
            {records.map((r) => (
              <Card key={r.id} hover className="h-full">
                <div className="flex flex-col gap-5 sm:flex-row">
                  <div className="shrink-0">
                    <QRCodeBox seed={r.qr_nfc_id} size={112} />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-xs uppercase tracking-[0.2em] text-gold">{r.zotaix_id}</p>
                    <p className="font-display mt-1 text-lg text-porcelain">{r.object_name}</p>
                    <p className="mt-1 text-xs text-mist">
                      {r.object_type.replace(/_/g, " ")}
                      {r.relationship_scene ? ` · ${r.relationship_scene}` : ""}
                    </p>
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <StatusPill status={r.delivery_status} />
                      <StatusPill status={r.aftercare_status} />
                      <Tag tone={r.privacy_level === "public" ? "jade" : "default"}>
                        {r.privacy_level === "public" ? (zh ? "公开" : "public") : zh ? "私密" : "private"}
                      </Tag>
                    </div>
                    {r.emotion_tags.length > 0 && (
                      <div className="mt-2.5 flex flex-wrap gap-2">
                        {r.emotion_tags.map((t) => (
                          <Tag key={t}>{t}</Tag>
                        ))}
                      </div>
                    )}
                    <div className="mt-4">
                      <Link href={`/reserve/${r.id}`} className="text-sm text-gold hover:underline">
                        {zh ? "打开证书页 →" : "Open certificate →"}
                      </Link>
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        ) : (
          <div className="mt-8">
            <EmptyState
              title={zh ? "档案馆正等待你的第一个对象" : "The archive is waiting for your first object"}
              description={
                zh
                  ? "先和 AI 礼宾创造一个对象并保存为草稿，然后回到这里把它入档为终身记录。"
                  : "Create an object with the AI concierge, save it as a draft, then return here to archive it as a lifetime record."
              }
              action={
                <ButtonLink href="/concierge" variant="gold">
                  {zh ? "创造第一个对象" : "Create your first object"}
                </ButtonLink>
              }
            />
          </div>
        )}

        {!user && (
          <div className="mt-8">
            <EmptyState
              title={zh ? "你的档案从注册开始" : "Your own archive starts with an account"}
              description={
                zh
                  ? "注册后，你生成并保存的每个对象都可以入档：获得 ZOTAIX 编号、QR/NFC 绑定与证书页面。"
                  : "Once registered, every object you generate and save can be archived — earning a ZOTAIX ID, a QR/NFC binding, and a certificate page."
              }
              action={
                <ButtonLink href="/register" variant="gold">
                  {zh ? "注册账号" : "Create an account"}
                </ButtonLink>
              }
            />
          </div>
        )}
      </Section>

      {/* Drafts awaiting archive */}
      {user && unarchivedDrafts.length > 0 && (
        <div className="border-y border-hairline bg-obsidian/40">
          <Section className="py-14 sm:py-20">
            <SectionHeader
              eyebrow={zh ? "待入档" : "Awaiting archive"}
              title={zh ? "这些草稿还没有身份" : "These drafts have no identity yet"}
              description={
                zh
                  ? "入档是免费的数字动作：草稿获得 ZOTAIX 编号、QR/NFC 标识与证书页。它是否成为实体，之后再决定。"
                  : "Archiving is a free, digital act: the draft receives a ZOTAIX ID, a QR/NFC mark, and a certificate page. Whether it becomes physical is a later decision."
              }
            />
            <div className="mt-8 grid gap-4 lg:grid-cols-2">
              {unarchivedDrafts.map((d) => (
                <Card key={d.id} className="h-full">
                  <div className="flex flex-wrap items-center gap-2">
                    <Tag tone="gold">{d.object_type.replace(/_/g, " ")}</Tag>
                    <StatusPill status={d.status} />
                  </div>
                  <p className="font-display mt-3 text-lg text-porcelain">{d.title}</p>
                  {d.label_copy && (
                    <blockquote className="mt-2 border-l-2 border-gold/50 pl-3 text-sm italic leading-relaxed text-mist">
                      “{d.label_copy}”
                    </blockquote>
                  )}
                  {d.emotion_tags.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {d.emotion_tags.map((t) => (
                        <Tag key={t}>{t}</Tag>
                      ))}
                    </div>
                  )}
                  <div className="mt-5">
                    <ArchiveActions draftId={d.id} zh={zh} />
                  </div>
                </Card>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* What a record contains */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "记录结构" : "Anatomy of a record"}
          title={zh ? "一条 Reserve 记录包含什么" : "What a Reserve record contains"}
          description={
            zh
              ? "档案不是订单回执，而是对象的完整身份：从情绪来源到生产批次，二十余个字段共同构成它的一生。"
              : "A record is not an order receipt — it is the object's full identity. More than twenty fields together tell the story of its life, from emotional origin to production batch."
          }
        />
        <div className="mt-10 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {RECORD_FIELDS.map((f) => (
            <div key={f.en} className="rounded-lg border border-hairline bg-obsidian/40 px-4 py-3">
              <p className="text-sm font-medium text-porcelain">{zh ? f.zh : f.en}</p>
              <p className="mt-1 text-xs leading-relaxed text-mist">{zh ? f.hintZh : f.hintEn}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* Aftercare & replenishment */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "售后与补铸" : "Aftercare & replenishment"}
            title={zh ? "档案在交付之后才真正开始工作" : "The record starts working after delivery"}
            description={
              zh
                ? "Reserve 是长期档案与复购的核心：售后与补铸附着在记录上，而不是附着在瓶子上。"
                : "Reserve is the long-term archive and the core of repeat casting: aftercare and replenishment attach to the record, not to the bottle."
            }
          />
          <div className="mt-10 grid gap-4 lg:grid-cols-3">
            <Card>
              <p className="text-lg text-gold">◈</p>
              <p className="font-display mt-2 text-base text-porcelain">{zh ? "售后时间线" : "Aftercare timeline"}</p>
              <p className="mt-2 text-sm leading-relaxed text-mist">
                {zh
                  ? "实体交付的对象带有生效中的售后状态：储存建议、瓶身养护、礼宾答疑。数字档案则以「无」标记——它们不需要养护，只需要被记得。"
                  : "Physically delivered objects carry an active aftercare status: storage guidance, bottle care, concierge Q&A. Digital-only records are marked “none” — they need no care, only remembering."}
              </p>
              <div className="mt-3 flex flex-wrap gap-2">
                <StatusPill status="active" />
                <StatusPill status="expired" />
                <StatusPill status="none" />
              </div>
            </Card>
            <Card>
              <p className="text-lg text-gold">▣</p>
              <p className="font-display mt-2 text-base text-porcelain">{zh ? "补铸入口" : "Replenishment entry"}</p>
              <p className="mt-2 text-sm leading-relaxed text-mist">
                {zh
                  ? "可补铸的记录保存了完整的配方方向、视觉与批次参数。在证书页上发起补铸，同一个对象可以在下一个纪念日被再次注满。"
                  : "Replenishable records keep the full direction, visual, and batch parameters. Start a replenishment from the certificate page and the same object can be poured again for the next anniversary."}
              </p>
              <p className="mt-3 text-xs text-mist">
                {zh
                  ? "补铸请求通过 Trade 通道提交，由人工礼宾审核数量、地区与合规后报价。"
                  : "Replenishment requests travel through the Trade channel — a human concierge reviews quantity, region, and compliance before quoting."}
              </p>
            </Card>
            <Card>
              <p className="text-lg text-gold">◉</p>
              <p className="font-display mt-2 text-base text-porcelain">{zh ? "QR / NFC 绑定" : "QR / NFC binding"}</p>
              <p className="mt-2 text-sm leading-relaxed text-mist">
                {zh
                  ? "每条记录生成唯一的 QR/NFC 标识。扫描实体上的标识即打开证书页——收到礼物的人读到的不是商品页，而是一段被写下来的时刻。"
                  : "Every record generates a unique QR/NFC identity. Scanning the mark on the physical object opens the certificate page — the recipient reads a written moment, not a product page."}
              </p>
              <p className="mt-3 text-xs text-mist">
                {zh ? "公开档案可分享；私密档案仅本人可见。" : "Public records are shareable; private records stay visible to you alone."}
              </p>
            </Card>
          </div>
        </Section>
      </div>

      {/* Repurchase logic */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "复铸流程" : "How replenishment works"}
          title={zh ? "从证书页到再次注满，四步" : "From certificate to a second pour, in four steps"}
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[
            {
              en: "Open the certificate",
              zh: "打开证书页",
              descEn: "Scan the QR/NFC mark or open the record from your archive.",
              descZh: "扫描 QR/NFC 标识，或从档案馆打开记录。",
            },
            {
              en: "Request replenishment",
              zh: "发起补铸",
              descEn: "One button on the certificate submits a replenishment request carrying the ZOTAIX ID.",
              descZh: "证书页上的补铸按钮会携带 ZOTAIX 编号提交请求。",
            },
            {
              en: "Human concierge review",
              zh: "人工礼宾审核",
              descEn: "Quantity, region, age compliance, and batch feasibility are confirmed before any quotation.",
              descZh: "数量、地区、年龄合规与批次可行性确认后才会报价。",
            },
            {
              en: "Deliver & update the archive",
              zh: "交付并更新档案",
              descEn: "The new pour joins the same record — one identity, a growing timeline.",
              descZh: "新一次铸造写入同一条记录——一个身份，一条不断延长的时间线。",
            },
          ].map((s, i) => (
            <Card key={s.en}>
              <p className="font-display text-3xl text-gold/60">{String(i + 1).padStart(2, "0")}</p>
              <p className="font-display mt-2 text-base text-porcelain">{zh ? s.zh : s.en}</p>
              <p className="mt-2 text-sm leading-relaxed text-mist">{zh ? s.descZh : s.descEn}</p>
            </Card>
          ))}
        </div>
        <div className="mt-8">
          <Notice tone="gold" title={zh ? "合规声明" : "Compliance"}>
            {pick(locale, complianceNotice)}{" "}
            {zh
              ? "补铸涉及酒类实体交付时，同样适用年龄与地区审核。"
              : "When a replenishment involves physical alcohol delivery, the same age and region checks apply."}
          </Notice>
        </div>
      </Section>

      {/* Membership entry */}
      <div className="border-t border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <Meridian className="mb-10" />
          <div className="grid items-center gap-8 lg:grid-cols-[1fr_auto]">
            <div>
              <SectionHeader
                eyebrow={zh ? "核心序列" : "Core Sequence"}
                title={zh ? "会员让档案馆更深" : "Membership makes the archive deeper"}
                description={
                  zh
                    ? "核心序列成员可以将记录封存为私密档案、优先进入补铸与生产排期、导出证书卡片，并为共创项目解锁发起资格。"
                    : "Core Sequence members can seal records as private, take priority in replenishment and production scheduling, export certificate cards, and unlock the right to start co-creation projects."
                }
              />
              <div className="mt-5 flex flex-wrap gap-2">
                <Tag tone="gold">{zh ? "私密档案" : "Private records"}</Tag>
                <Tag tone="gold">{zh ? "补铸优先" : "Replenishment priority"}</Tag>
                <Tag tone="gold">{zh ? "证书导出" : "Certificate export"}</Tag>
                <Tag tone="gold">{zh ? "共创发起权" : "Co-creation founding"}</Tag>
              </div>
            </div>
            <ButtonLink href="/membership" variant="gold">
              {zh ? "进入核心序列" : "Enter the Core Sequence"}
            </ButtonLink>
          </div>
        </Section>
      </div>
    </>
  );
}
