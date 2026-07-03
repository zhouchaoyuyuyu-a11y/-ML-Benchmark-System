import Link from "next/link";
import type { Metadata } from "next";
import AgeGate from "@/components/AgeGate";
import QRCodeBox from "@/components/QRCodeBox";
import { ButtonLink, Card, DefinitionRow, Notice, PageHero, Section, SectionHeader, Stat, Tag } from "@/components/ui";
import MaisonForms from "./MaisonForms";
import { complianceNotice, pick } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Maison ZOTAIX — enterprise gifting and private bespoke, confirmed by human concierges",
  description:
    "The premium line: AI-drafted proposals refined by human concierges into enterprise gifts, brand collaborations, and private bespoke objects — with quotation, sample paths, delivery, aftercare, and Reserve identity for every unit.",
  path: "/maison",
  keywords: ["enterprise gifting", "bespoke spirits", "brand collaboration", "private concierge", "Maison ZOTAIX"],
});

export default async function MaisonPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const s = data.settings;
  const conciergeHours = data.cms_blocks.find((b) => b.key === "maison.concierge.hours" && b.enabled);
  const meridianCase = data.case_studies.find((c) => c.slug === "meridian-hotels-vip");

  const audiences: { en: string; zh: string }[] = [
    { en: "Enterprise gifting", zh: "企业礼赠" },
    { en: "Client appreciation", zh: "客户答谢" },
    { en: "Banquets", zh: "高端宴席" },
    { en: "Private celebrations", zh: "私人庆典" },
    { en: "Hotels & clubs", zh: "酒店与会所" },
    { en: "Brand collaborations", zh: "品牌联名" },
    { en: "City souvenirs", zh: "城市伴手礼" },
    { en: "Cultural tourism", zh: "文旅项目" },
    { en: "Weddings", zh: "婚礼" },
    { en: "Anniversaries", zh: "周年纪念" },
    { en: "Collections", zh: "收藏" },
  ];

  const coreValues: { icon: string; en: string; zh: string }[] = [
    { icon: "◈", en: "AI concierge", zh: "AI 礼宾" },
    { icon: "◆", en: "Human confirmation", zh: "人工确认" },
    { icon: "❖", en: "Premium design", zh: "高级设计" },
    { icon: "▣", en: "Quotation", zh: "报价" },
    { icon: "⬡", en: "Delivery", zh: "交付" },
    { icon: "✦", en: "Aftercare", zh: "售后" },
    { icon: "◉", en: "Reserve identity", zh: "档案身份" },
  ];

  const thresholds: { qty: string; en: string; zh: string }[] = [
    { qty: `${s.co_create_public_threshold}`, en: "A public co-creation page opens", zh: "开启公开共创页" },
    { qty: `${s.co_create_review_threshold}`, en: "Human review of the concept", zh: "概念进入人工评审" },
    { qty: `${s.co_create_label_threshold}`, en: "Label & gift-box theme customization", zh: "解锁标签与礼盒主题定制" },
    { qty: `${s.co_create_flavor_threshold}`, en: "Flavor-direction review", zh: "解锁风味方向评审" },
    { qty: `${s.co_create_enterprise_threshold}`, en: "Enterprise gifting review", zh: "进入企业礼赠评审" },
    { qty: `${s.co_create_supply_threshold}+`, en: "Packaging & supply-chain customization", zh: "解锁包装与供应链定制" },
    { qty: `${s.co_create_partner_threshold}+`, en: "Partnership program", zh: "进入合作伙伴计划" },
  ];

  return (
    <>
      <AgeGate zh={zh} />

      <PageHero
        tone="maison"
        eyebrow="Maison ZOTAIX"
        title={zh ? "由 AI 起草，由人确认的高定礼赠" : "Premium gifting, drafted by AI, confirmed by humans"}
        description={
          zh
            ? "企业礼赠、品牌联名与私人高定的完整礼宾链路：AI 收集场景与预算并生成提案，人工礼宾确认设计、供应链与合规，随后是报价、打样、交付与售后——每一件交付物都绑定档案身份。"
            : "A complete concierge chain for enterprise gifting, brand collaborations, and private bespoke: AI gathers your scenario and budget into draft proposals; human concierges confirm design, supply chain, and compliance; then quotation, sampling, delivery, and aftercare — with a Reserve identity bound to every unit."
        }
      >
        <ButtonLink href="#enterprise" variant="gold">
          {zh ? "企业定制" : "Enterprise customization"}
        </ButtonLink>
        <ButtonLink href="#collaboration" variant="outline">
          {zh ? "品牌联名" : "Brand collaboration"}
        </ButtonLink>
        <ButtonLink href="#concierge" variant="outline">
          {zh ? "私人礼宾" : "Private concierge"}
        </ButtonLink>
      </PageHero>

      {/* Compliance — prominent */}
      <Section className="pt-8">
        <Notice tone="gold" title={zh ? "合规声明" : "Compliance"}>
          {pick(locale, complianceNotice)}{" "}
          {zh
            ? "酒类交付需通过年龄与地区合规审核；未成年人不适用酒类产品。"
            : "Alcohol deliveries pass age and region compliance checks; alcohol products are never offered to minors."}
        </Notice>
      </Section>

      {/* Audiences + core values */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "服务对象" : "Who Maison serves"}
          title={zh ? "十一种正式场合，一条礼宾链路" : "Eleven formal occasions, one concierge chain"}
        />
        <div className="mt-6 flex flex-wrap gap-2">
          {audiences.map((a) => (
            <Tag key={a.en} tone="gold">
              {zh ? a.zh : a.en}
            </Tag>
          ))}
        </div>
        <div className="mt-10 rounded-xl border border-hairline bg-obsidian/40">
          <div className="grid grid-cols-2 gap-px sm:grid-cols-4 lg:grid-cols-7">
            {coreValues.map((c) => (
              <div key={c.en} className="flex flex-col items-center gap-1.5 px-3 py-5 text-center">
                <span className="text-base text-gold">{c.icon}</span>
                <span className="text-xs text-porcelain">{zh ? c.zh : c.en}</span>
              </div>
            ))}
          </div>
        </div>
      </Section>

      {/* Case study */}
      {meridianCase && (
        <div className="border-y border-hairline bg-obsidian/40">
          <Section className="py-14 sm:py-20">
            <div className="grid gap-10 lg:grid-cols-2">
              <div>
                <SectionHeader
                  eyebrow={zh ? "已交付案例" : "A delivered program"}
                  title={zh ? meridianCase.title_zh : meridianCase.title}
                  description={meridianCase.summary}
                />
                <p className="mt-4 text-sm leading-relaxed text-mist">{meridianCase.story[2]}</p>
                <div className="mt-5">
                  <ButtonLink href={`/cases/${meridianCase.slug}`} variant="outline">
                    {zh ? "阅读完整案例 →" : "Read the full case →"}
                  </ButtonLink>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <Stat label={zh ? "交付数量" : "Units delivered"} value="300" hint={zh ? "分三城交付" : "Across three cities"} />
                <Stat label={zh ? "证书扫码率" : "Certificates scanned"} value="41%" hint={zh ? "交付后一个月内" : "Within one month"} />
                <Stat label={zh ? "样品路径" : "Sample path"} value="3" hint={zh ? "预产样先行确认" : "Pre-production samples first"} />
                <Stat label={zh ? "礼宾回复" : "Concierge reply"} value={zh ? "1 个工作日" : "1 business day"} hint={zh ? "首次响应时间" : "First response time"} />
              </div>
            </div>
          </Section>
        </div>
      )}

      {/* Enterprise customization */}
      <Section id="enterprise" className="scroll-mt-24 py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "企业定制" : "Enterprise customization"}
          title={zh ? "从场景与预算，到报价与交付" : "From scenario and budget to quotation and delivery"}
          description={
            zh
              ? "填写场景、数量、预算与四个设计方向，人工礼宾会核对可行性与合规，再给出正式报价与样品路径。方向写到什么程度都可以——礼宾会替你补全。"
              : "Describe the scenario, quantity, budget, and four design directions. A human concierge verifies feasibility and compliance, then returns a formal quotation and a sample path. Directions can be as rough as you like — the concierge completes them with you."
          }
        />
        <div className="mt-8 grid gap-5 lg:grid-cols-[1fr_320px]">
          <MaisonForms zh={zh} form="enterprise" />
          <div className="space-y-5">
            <Card className="border-gold/25">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
                {zh ? "先要一份 AI 草案？" : "Want an AI draft first?"}
              </p>
              <p className="mt-2 text-sm leading-relaxed text-mist">
                {zh
                  ? "在提交需求之前，可以让 AI 礼宾先生成一份企业提案草案——液体、香氛、瓶身与礼盒故事的初稿，带着草案来谈会更快。"
                  : "Before you submit, let the AI concierge draft an enterprise proposal — a first pass at liquid, fragrance, bottle, and gift-box story. Arriving with a draft makes the conversation faster."}
              </p>
              <div className="mt-4">
                <ButtonLink href="/forge?mode=enterprise" variant="gold">
                  {zh ? "生成企业提案草案" : "Generate an enterprise draft"}
                </ButtonLink>
              </div>
            </Card>
            <Card>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
                {zh ? "提交之后" : "After you submit"}
              </p>
              <ol className="mt-3 space-y-3">
                {[
                  { en: "Human review of quantity, budget, deadline, and region compliance", zh: "人工审核数量、预算、周期与地区合规" },
                  { en: "Design directions confirmed with the atelier and supply chain", zh: "设计方向与工坊、供应链逐项确认" },
                  { en: "Formal quotation and sample path proposal", zh: "正式报价与样品路径方案" },
                  { en: "Production, delivery, aftercare, and Reserve binding", zh: "生产、交付、售后与档案绑定" },
                ].map((step, i) => (
                  <li key={step.en} className="flex items-start gap-3 text-sm text-mist">
                    <span className="font-display text-gold/70">{String(i + 1).padStart(2, "0")}</span>
                    <span>{zh ? step.zh : step.en}</span>
                  </li>
                ))}
              </ol>
            </Card>
          </div>
        </div>
      </Section>

      {/* Thresholds */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "规模与深度" : "Scale and depth"}
            title={zh ? "数量越大，定制越深" : "Deeper customization unlocks with scale"}
            description={
              zh
                ? "单瓶定制建立在标准基液之上，承载你的表达；更大的批量按阶段解锁更深的定制层级——每一层都经人工评审。"
                : "Single-unit customization carries your expression on a standard base liquid; larger runs unlock deeper customization in stages — every stage reviewed by humans."
            }
          />
          <div className="mt-8 overflow-x-auto">
            <table className="w-full min-w-[480px] border-collapse text-left">
              <thead>
                <tr className="border-b border-hairline">
                  <th className="py-3 pr-6 text-xs font-semibold uppercase tracking-wider text-mist">
                    {zh ? "人数 / 数量" : "People / units"}
                  </th>
                  <th className="py-3 text-xs font-semibold uppercase tracking-wider text-mist">
                    {zh ? "解锁内容" : "What unlocks"}
                  </th>
                </tr>
              </thead>
              <tbody>
                {thresholds.map((t) => (
                  <tr key={t.qty} className="border-b border-hairline last:border-0">
                    <td className="py-3 pr-6 font-display text-lg text-gold">{t.qty}</td>
                    <td className="py-3 text-sm text-porcelain">{zh ? t.zh : t.en}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-xs text-mist">
            {zh
              ? "阈值同样适用于共创池项目——集体的规模可以把一个想法一路推进到企业级评审。"
              : "The same thresholds apply to co-creation pool projects — collective scale can carry an idea all the way to enterprise review."}
          </p>
        </Section>
      </div>

      {/* Brand collaboration */}
      <Section id="collaboration" className="scroll-mt-24 py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "品牌联名" : "Brand collaboration"}
          title={zh ? "把一座城市、一场展、一间酒店装进瓶子" : "Bottle a city, an exhibition, a hotel"}
          description={
            zh
              ? "城市伴手礼、文旅联名、设计周限定、酒店客房香氛线——留下品牌与场景，礼宾团队负责方向、可行性与合作框架。"
              : "City souvenirs, cultural-tourism editions, design-week runs, hotel amenity fragrance lines — leave your brand and scenario, and the concierge team handles direction, feasibility, and the collaboration framework."
          }
        />
        <div className="mt-8 max-w-3xl">
          <MaisonForms zh={zh} form="collaboration" />
        </div>
      </Section>

      {/* Private concierge */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section id="concierge" className="scroll-mt-24 py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "私人礼宾" : "Private concierge"}
            title={zh ? "一个人的重要时刻，也值得整条链路" : "One person's occasion deserves the whole chain"}
            description={
              zh
                ? "婚礼、周年、收藏、致敬——私人委托与企业项目走同一条人工礼宾链路：确认、设计、报价、交付与档案。"
                : "Weddings, anniversaries, collections, tributes — private commissions travel the same human-concierge chain as enterprise programs: confirmation, design, quotation, delivery, and archive."
            }
          />
          {conciergeHours && (
            <p className="mt-4 text-sm text-gold">◈ {conciergeHours.content}</p>
          )}
          <div className="mt-8 max-w-3xl">
            <MaisonForms zh={zh} form="concierge" />
          </div>
        </Section>
      </div>

      {/* Reserve binding */}
      <Section className="py-14 sm:py-20">
        <div className="grid gap-10 lg:grid-cols-[1fr_auto] lg:items-start">
          <div>
            <SectionHeader
              eyebrow={zh ? "档案身份" : "Reserve identity"}
              title={zh ? "每一件 Maison 交付物，都有一条档案" : "Every Maison delivery carries a Reserve record"}
              description={
                zh
                  ? "瓶会空，香会散；它为何存在的记录会留下来。Maison 交付物在出厂时即完成档案绑定："
                  : "Bottles empty and fragrances fade; the record of why they existed remains. Every Maison unit is bound to its archive before it ships:"
              }
            />
            <dl className="mt-6">
              <DefinitionRow label={zh ? "ZOTAIX ID" : "ZOTAIX ID"}>
                {zh
                  ? "唯一编号（如 ZX-2026-0611-0001），逐件签发，批次可溯。"
                  : "A unique serial (e.g. ZX-2026-0611-0001), issued per unit and traceable to its batch."}
              </DefinitionRow>
              <DefinitionRow label={zh ? "QR / NFC 绑定" : "QR / NFC binding"}>
                {zh
                  ? "收礼人扫码即见为其而写的故事页——不是商品页，而是一段被保存的时刻。"
                  : "A recipient scans the code and reads the story written for them — not a product page, but a kept moment."}
              </DefinitionRow>
              <DefinitionRow label={zh ? "证书页" : "Certificate page"}>
                {zh
                  ? "记录情绪标签、设计方向、版本与批次；公开或私密由委托方决定。"
                  : "Emotion tags, design directions, version, and batch — public or private, at the commissioner's choice."}
              </DefinitionRow>
              <DefinitionRow label={zh ? "售后与补铸" : "Aftercare & replenishment"}>
                {zh
                  ? "售后与补铸权益附着在档案上：明年的答谢、下一季的补货，从同一条记录再次出发。"
                  : "Aftercare and replenishment attach to the record itself — next year's appreciation and next season's refill start from the same line in the archive."}
              </DefinitionRow>
            </dl>
            <div className="mt-6">
              <ButtonLink href="/reserve" variant="outline">
                {zh ? "了解档案馆 →" : "Explore Reserve →"}
              </ButtonLink>
            </div>
          </div>
          <div className="flex justify-center lg:pt-10">
            <QRCodeBox seed="ZX-2026-0611-0001" label={zh ? "档案证书示例 · ZX-2026-0611-0001" : "Sample certificate · ZX-2026-0611-0001"} size={200} />
          </div>
        </div>
        <div className="mt-10">
          <Notice tone="gold">
            {zh
              ? "Maison 交付基于标准基液与既有瓶型模具承载个性化表达；我们不承诺逐瓶新配方或新模具，全部生产环节均经人工与供应链确认。"
              : "Maison deliveries carry personalized expression on standard base liquids and existing bottle molds; we never promise per-bottle new formulas or molds, and every production step is confirmed by humans and the supply chain."}
          </Notice>
        </div>
      </Section>

      {/* Footer link row */}
      <div className="border-t border-hairline bg-obsidian/40">
        <Section className="py-10">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <p className="text-sm text-mist">
              {zh
                ? "还在轻量场景？情绪补给线更适合日常时刻。"
                : "Working at a lighter scale? The emotional supply line suits everyday moments."}
            </p>
            <div className="flex flex-wrap gap-3">
              <Link href="/supply" className="text-sm text-gold hover:underline">
                {zh ? "前往 ZOTAIX Supply →" : "Go to ZOTAIX Supply →"}
              </Link>
              <Link href="/cases" className="text-sm text-gold hover:underline">
                {zh ? "查看全部案例 →" : "All cases →"}
              </Link>
            </div>
          </div>
        </Section>
      </div>
    </>
  );
}
