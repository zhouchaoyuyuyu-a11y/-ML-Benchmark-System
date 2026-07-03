import type { Metadata } from "next";
import AgeGate from "@/components/AgeGate";
import { ButtonLink, Card, EmptyState, Notice, PageHero, Section, SectionHeader, StatusPill, Tag } from "@/components/ui";
import TradeClient, { type TradeDraftOption } from "./TradeClient";
import { getSessionUser } from "@/lib/auth";
import { complianceNotice, pick } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Trade — quotes, creative rights, and enterprise inquiries, reviewed by humans",
  description:
    "ZOTAIX Trade handles creative-rights authorization, small-batch casting quotes, co-creation founder rights, enterprise inquiries, and replenishment. Every request passes human review and compliance checks before quotation — this is not an alcohol resale marketplace.",
  path: "/trade",
  keywords: ["creative rights", "authorization", "small-batch quote", "enterprise gifting inquiry", "replenishment"],
});

export default async function TradePage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const user = await getSessionUser();

  const myDrafts: TradeDraftOption[] = user
    ? data.object_drafts
        .filter((d) => d.user_id === user.id)
        .map((d) => ({ id: d.id, title: d.title, object_type: d.object_type, budget: d.budget }))
    : [];
  const myRequests = user ? data.trade_requests.filter((t) => t.user_id === user.id) : [];

  const handles: { icon: string; en: string; zh: string; descEn: string; descZh: string }[] = [
    {
      icon: "❖",
      en: "Creative-rights authorization",
      zh: "创意授权",
      descEn: "Authorize a saved design — label copy, visual style, naming — for commercial use under a reviewed agreement with an agreed income share for the creator.",
      descZh: "将已保存的设计——瓶身文案、视觉风格、命名——在经审核的协议下授权商用，创作者按约定获得收益分成。",
    },
    {
      icon: "◈",
      en: "Small-batch casting quotes",
      zh: "小批量铸造报价",
      descEn: "Quotations for 12–300 units on a standard, quality-controlled base liquid carrying your expression. A human concierge confirms every direction before pricing.",
      descZh: "12–300 件的小批量报价：标准品控基液承载你的表达，人工礼宾在定价前确认每一项方向。",
    },
    {
      icon: "⬡",
      en: "Co-creation founder rights",
      zh: "共创发起人权益",
      descEn: "Founder slots, serials, and engraving rights for co-creation runs — routed through project review and bound to the project's Reserve archive.",
      descZh: "共创批次的发起人名额、编号与镌刻权益——经项目评审后绑定共创档案。",
    },
    {
      icon: "◆",
      en: "Enterprise inquiries",
      zh: "企业询价",
      descEn: "Gifting programs with invoices, sample paths, staged multi-city delivery, and label compliance — refined with a dedicated concierge.",
      descZh: "含发票、样品路径、多城分批交付与标签合规的企业礼赠项目，由专属礼宾跟进。",
    },
    {
      icon: "▣",
      en: "Replenishment",
      zh: "补铸",
      descEn: "Re-order a delivered object directly from its Reserve record — same expression, fresh batch, renewed compliance check.",
      descZh: "从档案记录直接补铸已交付的对象——同一表达、全新批次、重新合规审核。",
    },
  ];

  const prohibited: { en: string; zh: string }[] = [
    { en: "User-to-user alcohol resale of any kind", zh: "任何形式的用户间酒类转售" },
    { en: "Alcohol vouchers or credits that have not passed platform review", zh: "未经平台审核的酒类券或额度" },
    { en: "Unauthorized third-party sellers operating under ZOTAIX designs", zh: "冒用 ZOTAIX 设计的未授权第三方销售" },
    { en: "Platform-external private transactions between users", zh: "用户之间的平台外私下交易" },
    { en: "Unreviewed spot trading of bottles, batches, or founder slots", zh: "未经审核的瓶身、批次或发起人名额现货交易" },
  ];

  const workflow: { en: string; zh: string; descEn: string; descZh: string }[] = [
    { en: "Submit", zh: "提交", descEn: "Attach a saved draft with quantity, budget, deadline, and region.", descZh: "附上已保存草稿与数量、预算、期限、地区。" },
    { en: "Human review", zh: "人工审核", descEn: "A concierge validates intent, feasibility, and design directions.", descZh: "礼宾核验意图、可行性与设计方向。" },
    { en: "Compliance check", zh: "合规审查", descEn: "Age, region, label, and alcohol rules are verified per request.", descZh: "逐单核验年龄、地区、标签与酒类规则。" },
    { en: "Quotation", zh: "报价", descEn: "A formal quote arrives with the sample path and timeline.", descZh: "正式报价单送达，含样品路径与时间表。" },
    { en: "Confirmation", zh: "确认", descEn: "You accept, adjust, or decline — nothing proceeds without you.", descZh: "由你接受、调整或婉拒——没有你的确认不会推进。" },
    { en: "Delivery", zh: "交付", descEn: "Production and logistics run in reviewed, staged batches.", descZh: "生产与物流按已审核的分批节奏执行。" },
    { en: "Reserve binding", zh: "档案绑定", descEn: "Every delivered unit binds to a Reserve record with QR identity.", descZh: "每件交付物绑定带 QR 身份的档案记录。" },
  ];

  const confirmationFields: { field: { en: string; zh: string }; confirms: { en: string; zh: string }; owner: { en: string; zh: string } }[] = [
    { field: { en: "Quantity", zh: "数量" }, confirms: { en: "Unit count and batch structure — depth of customization scales with volume.", zh: "件数与批次结构——定制深度随数量提升。" }, owner: { en: "You + concierge", zh: "你 + 礼宾" } },
    { field: { en: "Budget", zh: "预算" }, confirms: { en: "Total or per-unit range that the quotation must respect.", zh: "报价必须遵守的总额或单件区间。" }, owner: { en: "You", zh: "你" } },
    { field: { en: "Deadline", zh: "交付期限" }, confirms: { en: "Delivery date checked against production and review lead times.", zh: "交付日期与生产及审核周期核对。" }, owner: { en: "Concierge", zh: "礼宾" } },
    { field: { en: "Delivery region", zh: "交付地区" }, confirms: { en: "Destination cities and their regional alcohol regulations.", zh: "目的城市及其地区性酒类法规。" }, owner: { en: "Compliance review", zh: "合规审核" } },
    { field: { en: "Liquid direction", zh: "液体方向" }, confirms: { en: "Base spirit, flavor, and proof — directions only; formulas are confirmed with the supply chain.", zh: "基酒、风味与度数——仅为方向，配方由供应链确认。" }, owner: { en: "Concierge + supply chain", zh: "礼宾 + 供应链" } },
    { field: { en: "Fragrance direction", zh: "香氛方向" }, confirms: { en: "Scent family and composition intent for fragrance objects.", zh: "香氛对象的香调家族与构成意图。" }, owner: { en: "Concierge + atelier", zh: "礼宾 + 工坊" } },
    { field: { en: "Bottle direction", zh: "瓶身方向" }, confirms: { en: "Silhouette, material, and finishing from the existing mold library.", zh: "从现有模具库确认造型、材质与表面处理。" }, owner: { en: "Concierge + atelier", zh: "礼宾 + 工坊" } },
    { field: { en: "Packaging direction", zh: "包装方向" }, confirms: { en: "Gift-box structure, materials, and printing techniques.", zh: "礼盒结构、材料与印刷工艺。" }, owner: { en: "Concierge + atelier", zh: "礼宾 + 工坊" } },
    { field: { en: "Sample path", zh: "样品路径" }, confirms: { en: "Whether pre-production samples are needed, how many, and by when.", zh: "是否需要预产样品、数量与时间点。" }, owner: { en: "You + concierge", zh: "你 + 礼宾" } },
    { field: { en: "Invoice", zh: "发票" }, confirms: { en: "Invoicing entity and documentation for enterprise settlement.", zh: "企业结算所需的开票主体与文件。" }, owner: { en: "Concierge", zh: "礼宾" } },
    { field: { en: "Logistics", zh: "物流" }, confirms: { en: "Staged delivery, temperature control, and multi-city routing.", zh: "分批交付、温控与多城路线。" }, owner: { en: "Concierge + logistics", zh: "礼宾 + 物流" } },
    { field: { en: "Label compliance", zh: "标签合规" }, confirms: { en: "Label copy reviewed against advertising and labeling regulations.", zh: "瓶身文案按广告与标签法规审核。" }, owner: { en: "Compliance review", zh: "合规审核" } },
    { field: { en: "Alcohol compliance", zh: "酒类合规" }, confirms: { en: "Age and region checks — alcohol is never delivered to minors.", zh: "年龄与地区审核——绝不向未成年人交付酒类。" }, owner: { en: "Compliance review", zh: "合规审核" } },
    { field: { en: "Aftercare", zh: "售后" }, confirms: { en: "Aftercare window, replenishment eligibility, and support channel.", zh: "售后周期、补铸资格与支持渠道。" }, owner: { en: "Concierge", zh: "礼宾" } },
    { field: { en: "Reserve record", zh: "档案记录" }, confirms: { en: "Each delivered unit binds to a ZOTAIX ID with QR/NFC identity.", zh: "每件交付物绑定 ZOTAIX ID 与 QR/NFC 身份。" }, owner: { en: "Platform", zh: "平台" } },
  ];

  return (
    <>
      <AgeGate zh={zh} />

      <PageHero
        tone="maison"
        eyebrow={zh ? "ZOTAIX Trade · 报价与授权" : "ZOTAIX Trade · Quotes & Rights"}
        title={zh ? "把已保存的对象，交给人工礼宾报价" : "Hand your saved object to a human concierge for quotation"}
        description={
          zh
            ? "Trade 是 ZOTAIX 的报价与授权通道：创意授权、小批量铸造报价、共创发起人权益、企业询价与补铸，全部经人工审核与合规审查。Trade 不是酒类转售市场——平台上不存在用户间的酒类买卖。"
            : "Trade is the quotation and rights channel of ZOTAIX: creative-rights authorization, small-batch casting quotes, co-creation founder rights, enterprise inquiries, and replenishment — every request passes human review and compliance checks. Trade is not an alcohol resale marketplace: no user-to-user alcohol transactions exist on this platform."
        }
      >
        <ButtonLink href="#quote" variant="gold">
          {zh ? "提交报价请求" : "Request a quote"}
        </ButtonLink>
        <ButtonLink href="/market" variant="outline">
          {zh ? "浏览创意市场" : "Browse the creative market"}
        </ButtonLink>
        <ButtonLink href="/maison#enterprise" variant="outline">
          {zh ? "企业询价" : "Enterprise inquiry"}
        </ButtonLink>
      </PageHero>

      <Section className="pt-8">
        <Notice tone="gold" title={zh ? "合规声明" : "Compliance"}>
          {pick(locale, complianceNotice)}{" "}
          {zh
            ? "酒类交付需通过年龄与地区合规审核。"
            : "Alcohol deliveries pass age and region compliance checks before any quotation is issued."}
        </Notice>
      </Section>

      {/* What Trade handles */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "Trade 处理什么" : "What Trade handles"}
          title={zh ? "五类请求，一条人工审核链" : "Five request types, one human review chain"}
          description={
            zh
              ? "每一类请求都始于一个已保存的对象或一个明确的场景，终于一份由人确认的报价与交付。"
              : "Every request begins with a saved object or a clear scenario, and ends with a quotation and delivery confirmed by humans."
          }
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {handles.map((h) => (
            <Card key={h.en} className="h-full">
              <div className="flex items-start gap-3">
                <span className="text-lg text-gold">{h.icon}</span>
                <div>
                  <p className="font-display text-base text-porcelain">{zh ? h.zh : h.en}</p>
                  <p className="mt-1.5 text-sm leading-relaxed text-mist">{zh ? h.descZh : h.descEn}</p>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </Section>

      {/* What Trade prohibits */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "Trade 禁止什么" : "What Trade prohibits"}
            title={zh ? "不是转售市场，也永远不会是" : "Not a resale marketplace — and never becoming one"}
          />
          <div className="mt-8">
            <Notice tone="ember" title={zh ? "以下行为在平台上被禁止" : "The following are prohibited on this platform"}>
              <ul className="mt-1 space-y-2">
                {prohibited.map((p) => (
                  <li key={p.en} className="flex items-start gap-2.5">
                    <span className="mt-0.5 text-ember">✕</span>
                    <span>{zh ? p.zh : p.en}</span>
                  </li>
                ))}
              </ul>
              <p className="mt-3 text-xs">
                {zh
                  ? "所有实体执行必须经过平台人工审核。检测到的违规行为会进入风控日志并可能导致账户限制。"
                  : "All physical execution must pass platform human review. Detected violations enter the moderation log and may lead to account restrictions."}
              </p>
            </Notice>
          </div>
        </Section>
      </div>

      {/* Quote workflow */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "报价流程" : "Quote workflow"}
          title={zh ? "七步：从提交到档案绑定" : "Seven steps: from submission to Reserve binding"}
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {workflow.map((w, i) => (
            <Card key={w.en} className="relative h-full">
              <p className="font-display text-2xl text-gold/60">{String(i + 1).padStart(2, "0")}</p>
              <p className="font-display mt-1.5 text-base text-porcelain">{zh ? w.zh : w.en}</p>
              <p className="mt-1.5 text-sm leading-relaxed text-mist">{zh ? w.descZh : w.descEn}</p>
            </Card>
          ))}
        </div>
      </Section>

      {/* Confirmation fields table */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "确认清单" : "Confirmation checklist"}
            title={zh ? "报价前，这些字段逐项确认" : "Before quotation, every field is confirmed"}
            description={
              zh
                ? "人工礼宾与合规团队在出具报价前逐项确认以下内容——这份清单也是你收到的报价单的结构。"
                : "The human concierge and compliance team confirm each of these before a quote is issued — this checklist is also the structure of the quotation you receive."
            }
          />
          <div className="zx-card mt-8 overflow-x-auto p-0">
            <table className="w-full min-w-[680px] text-left text-sm">
              <thead>
                <tr className="border-b border-hairline text-xs uppercase tracking-wider text-mist">
                  <th className="px-4 py-3 font-medium sm:px-5">{zh ? "字段" : "Field"}</th>
                  <th className="px-4 py-3 font-medium sm:px-5">{zh ? "确认内容" : "What is confirmed"}</th>
                  <th className="px-4 py-3 font-medium sm:px-5">{zh ? "确认方" : "Confirmed by"}</th>
                </tr>
              </thead>
              <tbody>
                {confirmationFields.map((row) => (
                  <tr key={row.field.en} className="border-b border-hairline last:border-0">
                    <td className="whitespace-nowrap px-4 py-3 text-porcelain sm:px-5">{zh ? row.field.zh : row.field.en}</td>
                    <td className="px-4 py-3 leading-relaxed text-mist sm:px-5">{zh ? row.confirms.zh : row.confirms.en}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-xs text-gold sm:px-5">{zh ? row.owner.zh : row.owner.en}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      </div>

      {/* Quote request form */}
      <Section id="quote" className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "提交请求" : "Submit a request"}
          title={zh ? "为一份已保存的草稿申请报价" : "Request a quote for a saved draft"}
          description={
            zh
              ? "报价始终附着在一个已保存的对象上——选择你的草稿，补充数量、预算与交付信息，人工礼宾会在一个工作日内回复。"
              : "Quotes always attach to a saved object — pick one of your drafts, add quantity, budget, and delivery details, and a human concierge replies within one business day."
          }
        />
        <div className="mt-8 max-w-3xl">
          <TradeClient zh={zh} signedIn={!!user} drafts={myDrafts} />
        </div>
      </Section>

      {/* Existing trade requests */}
      <div className="border-t border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "你的请求" : "Your requests"}
            title={zh ? "交易请求与审核状态" : "Trade requests and their review states"}
            description={
              zh
                ? "每一条请求同时显示合规状态、人工审核状态与报价状态。"
                : "Each request shows its compliance status, human review status, and quote status side by side."
            }
          />
          <div className="mt-8">
            {!user ? (
              <Notice tone="gold" title={zh ? "登录后查看" : "Sign in to view"}>
                {zh
                  ? "登录后，这里会显示你提交过的所有报价、授权与补铸请求及其审核进度。"
                  : "Once signed in, this area lists every quote, authorization, and replenishment request you have submitted, with its live review progress."}{" "}
                <ButtonLink href="/login" variant="ghost" className="!px-0 text-gold">
                  {zh ? "去登录 →" : "Sign in →"}
                </ButtonLink>
              </Notice>
            ) : myRequests.length === 0 ? (
              <EmptyState
                title={zh ? "还没有交易请求" : "No trade requests yet"}
                description={
                  zh
                    ? "从上方表单为一份草稿申请报价，或先在 Forge 创造一个对象。"
                    : "Request a quote for a draft using the form above, or create an object in Forge first."
                }
                action={<ButtonLink href="/forge" variant="outline">{zh ? "去 Forge 创造" : "Create in Forge"}</ButtonLink>}
              />
            ) : (
              <div className="grid gap-4 lg:grid-cols-2">
                {myRequests.map((t) => {
                  const draft = t.object_draft_id ? data.object_drafts.find((d) => d.id === t.object_draft_id) : undefined;
                  return (
                    <Card key={t.id} className="h-full">
                      <div className="flex flex-wrap items-center gap-2">
                        <Tag tone="gold">{t.request_type}</Tag>
                        <span className="text-xs text-mist">{t.id}</span>
                        <span className="ml-auto text-xs text-mist">{t.created_at.slice(0, 10)}</span>
                      </div>
                      <p className="font-display mt-3 text-base text-porcelain">
                        {draft ? draft.title : zh ? "场景询价" : "Scenario inquiry"}
                      </p>
                      <div className="mt-2 grid gap-x-6 gap-y-1 text-xs text-mist sm:grid-cols-2">
                        <span>{zh ? "数量" : "Quantity"} · {t.quantity}</span>
                        <span>{zh ? "预算" : "Budget"} · {t.budget}</span>
                        {t.deadline && <span>{zh ? "期限" : "Deadline"} · {t.deadline}</span>}
                        {t.delivery_region && <span>{zh ? "地区" : "Region"} · {t.delivery_region}</span>}
                      </div>
                      <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2">
                        <span className="flex items-center gap-2 text-xs text-mist">
                          {zh ? "合规" : "Compliance"} <StatusPill status={t.compliance_status} />
                        </span>
                        <span className="flex items-center gap-2 text-xs text-mist">
                          {zh ? "人工审核" : "Human review"} <StatusPill status={t.human_review_status} />
                        </span>
                        <span className="flex items-center gap-2 text-xs text-mist">
                          {zh ? "报价" : "Quote"} <StatusPill status={t.quote_status} />
                        </span>
                      </div>
                      {t.notes && <p className="mt-3 text-xs leading-relaxed text-mist">{t.notes}</p>}
                    </Card>
                  );
                })}
              </div>
            )}
          </div>
        </Section>
      </div>
    </>
  );
}
