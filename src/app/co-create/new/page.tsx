import type { Metadata } from "next";
import { ButtonLink, Card, Notice, PageHero, Section, SectionHeader, Tag } from "@/components/ui";
import NewProjectClient from "./NewProjectClient";
import { getSessionUser } from "@/lib/auth";
import { complianceNotice, pick } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Start a co-creation project — founder rights for Core Sequence members",
  description:
    "Publish a generated concept to the ZOTAIX Co-Creation Pool: set a target quantity, keep the Founder Edition serial and engraving, and let the community carry it across the thresholds.",
  path: "/co-create/new",
  keywords: ["start co-creation", "founder rights", "creative casting", "Core Sequence"],
});

export default async function NewCoCreatePage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const s = data.settings;
  const user = await getSessionUser();
  const isMember = !!user && user.membership_level !== "free";

  const founderRights = [
    {
      icon: "◆",
      en: "Founder Edition serial",
      zh: "创始版序列号",
      descEn: "Every founder unit carries a dedicated serial and your engraved name.",
      descZh: "每一份创始版都带有专属序列号与你的镌刻署名。",
    },
    {
      icon: "▣",
      en: "Exclusive QR archive page",
      zh: "专属 QR 档案页",
      descEn: "The project's archive page opens with the founder's story — bound by QR for life.",
      descZh: "项目档案页以发起人的故事开篇——QR 终身绑定。",
    },
    {
      icon: "◈",
      en: "Founder digital mark",
      zh: "发起人数字印记",
      descEn: "A founder mark joins your profile and Reserve the moment the project passes review.",
      descZh: "项目过审即刻，发起人印记写入你的个人中心与档案馆。",
    },
    {
      icon: "❖",
      en: "Priority voice at reviews",
      zh: "评审优先话语权",
      descEn: "At each threshold review, the founder's direction leads label, flavor, and packaging decisions.",
      descZh: "每一级门槛评审中，发起人的方向主导标签、风味与包装决策。",
    },
  ];

  return (
    <>
      <PageHero
        eyebrow={zh ? "共创池 · 发起项目" : "Co-Creation Pool · start a project"}
        title={zh ? "把你的概念交给一群人铸造" : "Hand your concept to a pool of casters"}
        description={
          zh
            ? "发起是核心序列会员权益：你提供概念与首铸份数，平台负责审核与供应链，社区负责把它推过一级级门槛。发起人保留序列号、镌刻与专属档案页。"
            : "Founding is a Core Sequence membership benefit: you bring the concept and your founder units, the platform brings review and supply chain, and the community carries it across the thresholds. The founder keeps the serial, the engraving, and an exclusive archive page."
        }
      >
        <ButtonLink href="#new-project" variant="gold">
          {zh ? "填写项目提案" : "Draft the proposal"}
        </ButtonLink>
        <ButtonLink href="/co-create" variant="outline">
          {zh ? "先看看集结中的项目" : "See gathering projects first"}
        </ButtonLink>
      </PageHero>

      {/* Founder rights */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "发起人权益" : "Founder rights"}
          title={zh ? "发起人保留的四样东西" : "Four things the founder keeps"}
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {founderRights.map((r) => (
            <Card key={r.en}>
              <p className="text-lg text-gold">{r.icon}</p>
              <p className="font-display mt-2 text-base text-porcelain">{zh ? r.zh : r.en}</p>
              <p className="mt-2 text-sm leading-relaxed text-mist">{zh ? r.descZh : r.descEn}</p>
            </Card>
          ))}
        </div>
        <div className="mt-6 flex flex-wrap gap-2">
          <Tag tone="gold">
            {zh
              ? `${s.co_create_public_threshold} 人 → 公开页`
              : `${s.co_create_public_threshold} people → public page`}
          </Tag>
          <Tag tone="gold">
            {zh
              ? `${s.co_create_label_threshold} 份 → 标签共创`
              : `${s.co_create_label_threshold} units → label co-creation`}
          </Tag>
          <Tag tone="gold">
            {zh
              ? `${s.co_create_flavor_threshold} 份 → 风味评审`
              : `${s.co_create_flavor_threshold} units → flavor review`}
          </Tag>
          <Tag tone="gold">
            {zh
              ? `${s.co_create_enterprise_threshold} 份 → 企业通道`
              : `${s.co_create_enterprise_threshold} units → enterprise track`}
          </Tag>
        </div>
      </Section>

      {/* Form */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section id="new-project" className="scroll-mt-24 py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "项目提案" : "The proposal"}
            title={zh ? "六个字段，一个项目" : "Six fields, one project"}
            description={
              zh
                ? "写下标题与概念，选择产品类型与目标份数，加上情绪标签与你自己的首铸份数。提交后项目进入平台评审。"
                : "Write the title and concept, pick a product type and target quantity, add emotion tags and your own founder units. On submission the project enters platform review."
            }
          />
          <div className="mt-10 grid gap-8 lg:grid-cols-[1fr_340px]">
            <NewProjectClient zh={zh} signedIn={!!user} isMember={isMember} />
            <div className="space-y-6">
              <Card>
                <p className="font-display text-base text-porcelain">{zh ? "评审会看什么" : "What review looks at"}</p>
                <ul className="mt-3 space-y-2 text-sm leading-relaxed text-mist">
                  {[
                    { en: "Sensitive content & public display", zh: "敏感内容与公开展示" },
                    { en: "Alcohol compliance & minor safety", zh: "酒类合规与未成年人保护" },
                    { en: "Copyright of names and imagery", zh: "命名与视觉的版权" },
                    { en: "Feasibility with the supply chain", zh: "供应链可行性" },
                    { en: "Trade eligibility of the run", zh: "项目的交易资格" },
                  ].map((item) => (
                    <li key={item.en} className="flex items-start gap-2.5">
                      <span className="mt-0.5 text-gold">✓</span>
                      <span>{zh ? item.zh : item.en}</span>
                    </li>
                  ))}
                </ul>
                <p className="mt-3 text-xs leading-relaxed text-mist">
                  {zh
                    ? "评审通常在一个工作日内完成；通过后项目公开陈列并开放投票与加入。"
                    : "Review usually completes within one business day; approved projects go public with voting and joining open."}
                </p>
              </Card>
              <Notice tone="gold" title={zh ? "核心序列权益" : "A Core Sequence benefit"}>
                {zh
                  ? "发起项目属于核心序列（Lite / Pro）会员权益；所有注册用户都可以加入与投票。"
                  : "Starting a project belongs to Core Sequence (Lite / Pro) membership; every registered user can join and vote."}{" "}
                <ButtonLink href="/membership" variant="ghost" className="!px-0 text-gold">
                  {zh ? "了解核心序列 →" : "About the Core Sequence →"}
                </ButtonLink>
              </Notice>
            </div>
          </div>
        </Section>
      </div>

      <Section className="py-12 sm:py-16">
        <Notice tone="gold" title={zh ? "合规声明" : "Compliance"}>
          {pick(locale, complianceNotice)}{" "}
          {zh
            ? "共创项目的实体交付在达成后经人工确认、供应链确认与年龄地区审核推进。"
            : "Physical delivery of a completed run proceeds through human confirmation, supply-chain confirmation, and age and region checks."}
        </Notice>
      </Section>
    </>
  );
}
