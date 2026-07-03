import type { Metadata } from "next";
import {
  ButtonLink,
  Card,
  Notice,
  PageHero,
  Section,
  SectionHeader,
  StatusPill,
  Tag,
} from "@/components/ui";
import CoCreateListClient from "./CoCreateListClient";
import { complianceNotice, pick } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "Co-Creation Pool — turn one concept into a shared casting run",
  description:
    "The ZOTAIX Co-Creation Pool (Creative Casting Pool / Order Reconstruction Bureau): publish a generated concept, gather supporters, cross review thresholds, and unlock founder rights, label co-creation, and flavor review.",
  path: "/co-create",
  keywords: ["co-creation", "creative casting pool", "order reconstruction", "founder rights", "group casting"],
});

export default async function CoCreatePage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const s = data.settings;

  const approved = data.co_creation_projects.filter((p) => p.public_visible && p.review_status === "approved");
  const inReview = data.co_creation_projects.filter((p) => p.review_status === "pending");

  const flow = [
    {
      en: "Generate",
      zh: "生成",
      descEn: "Start from an AI-generated object: a liquid or fragrance direction, names, and one line of label copy.",
      descZh: "从 AI 生成的对象开始：酒体或香氛方向、命名与一句瓶身文案。",
    },
    {
      en: "Publish",
      zh: "发布",
      descEn: "Submit the concept to the pool. Platform review clears it for public listing.",
      descZh: "把概念提交到共创池，通过平台审核后公开陈列。",
    },
    {
      en: "Gather",
      zh: "集结",
      descEn: "The community browses, likes, saves, comments, votes — and joins with reserved quantities.",
      descZh: "社区浏览、点赞、收藏、评论、投票——并以预订数量加入。",
    },
    {
      en: "Cross thresholds",
      zh: "跨越门槛",
      descEn: "Each threshold unlocks a deeper stage: review, label co-creation, flavor review, enterprise track.",
      descZh: "每一级门槛解锁更深的阶段：评审、标签共创、风味评审、企业通道。",
    },
    {
      en: "Founder rights",
      zh: "发起人权益",
      descEn: "The founder keeps the serial, engraving, and an exclusive QR archive page.",
      descZh: "发起人保留序列号、镌刻与专属 QR 档案页。",
    },
    {
      en: "Participant benefits",
      zh: "参与者权益",
      descEn: "Every participant's unit binds to the shared batch archive with a personal digital mark.",
      descZh: "每位参与者的份额绑定共享批次档案，并获得个人数字印记。",
    },
    {
      en: "Trade & Reserve",
      zh: "交付与入档",
      descEn: "Production runs through Trade with human review; every delivered unit lands in Reserve.",
      descZh: "生产经 Trade 与人工审核推进；每一件交付都写入 Reserve 档案。",
    },
  ];

  const ladder = [
    {
      value: s.co_create_public_threshold,
      unitEn: "people",
      unitZh: "人",
      en: "Public project page opens",
      zh: "公开项目页开启",
      descEn: "Ten supporters give the concept a public page with voting and joining.",
      descZh: "十位支持者让概念获得可投票、可加入的公开页面。",
    },
    {
      value: s.co_create_review_threshold,
      unitEn: "units",
      unitZh: "份",
      en: "Platform review deepens",
      zh: "平台评审加深",
      descEn: "Compliance, feasibility, and supply checks run at production depth.",
      descZh: "合规、可行性与供应链检查进入生产级深度。",
    },
    {
      value: s.co_create_label_threshold,
      unitEn: "units",
      unitZh: "份",
      en: "Label & gift-box co-creation",
      zh: "标签与礼盒共创",
      descEn: "Participants co-decide label copy, palette, and gift-box theming.",
      descZh: "参与者共同决定文案、配色与礼盒主题。",
    },
    {
      value: s.co_create_flavor_threshold,
      unitEn: "units",
      unitZh: "份",
      en: "Flavor-direction review",
      zh: "风味方向评审",
      descEn: "The atelier reviews the flavor direction with the supply chain.",
      descZh: "工坊与供应链共同评审风味方向。",
    },
    {
      value: s.co_create_enterprise_threshold,
      unitEn: "units",
      unitZh: "份",
      en: "Enterprise gifting review",
      zh: "企业礼赠评审",
      descEn: "The run qualifies for the enterprise gifting track with a dedicated concierge.",
      descZh: "项目进入企业礼赠通道，由专属礼宾跟进。",
    },
    {
      value: s.co_create_supply_threshold,
      unitEn: "units+",
      unitZh: "份+",
      en: "Supply-chain scheduling",
      zh: "供应链排产",
      descEn: "A dedicated production window is scheduled with the supply chain.",
      descZh: "与供应链锁定专属生产窗口。",
    },
    {
      value: s.co_create_partner_threshold,
      unitEn: "units+",
      unitZh: "份+",
      en: "Brand partnership track",
      zh: "品牌合作通道",
      descEn: "The project graduates into a long-term brand partnership conversation.",
      descZh: "项目升级为长期品牌合作洽谈。",
    },
  ];

  return (
    <>
      <PageHero
        eyebrow={zh ? "共创铸造池 · Creative Casting Pool" : "Co-Creation Pool · Creative Casting Pool"}
        title={
          zh
            ? "一个人的概念，一群人的铸造"
            : "One person's concept, a whole pool's casting"
        }
        description={
          zh
            ? "共创池也被称为「创意铸造池」与「订单重构局」：它把单个订单重构为集体铸造。发布一个 AI 生成的概念，集结支持者，跨越门槛——十人开启公开页，百瓶解锁风味评审。"
            : "Also known as the Creative Casting Pool and the Order Reconstruction Bureau — the pool reconstructs single orders into collective castings. Publish an AI-generated concept, gather supporters, and cross thresholds: ten people open a public page, one hundred bottles unlock flavor review."
        }
      >
        <ButtonLink href="/co-create/new" variant="gold">
          {zh ? "发起一个项目" : "Start a project"}
        </ButtonLink>
        <ButtonLink href="#projects" variant="outline">
          {zh ? "浏览集结中的项目" : "Browse gathering projects"}
        </ButtonLink>
        <ButtonLink href="/membership" variant="outline">
          {zh ? "核心序列" : "Core Sequence"}
        </ButtonLink>
      </PageHero>

      {/* Flow 1–7 */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "共创流程" : "How the pool works"}
          title={zh ? "从生成到入档，七步" : "From generation to the archive, in seven steps"}
          description={
            zh
              ? "共创不是团购：它是一次有审核、有权益、有档案的集体创作。每一步都有明确的规则与人工把关。"
              : "Co-creation is not group buying — it is collective authorship with review, rights, and an archive. Every step carries explicit rules and human oversight."
          }
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {flow.map((f, i) => (
            <Card key={f.en}>
              <p className="font-display text-3xl text-gold/60">{String(i + 1).padStart(2, "0")}</p>
              <p className="font-display mt-2 text-base text-porcelain">{zh ? f.zh : f.en}</p>
              <p className="mt-2 text-sm leading-relaxed text-mist">{zh ? f.descZh : f.descEn}</p>
            </Card>
          ))}
        </div>
      </Section>

      {/* Threshold ladder */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "门槛阶梯" : "The threshold ladder"}
            title={zh ? "规模越深，解锁越深" : "Depth scales with commitment"}
            description={
              zh
                ? "定制深度按承诺规模分级解锁——这是小批量定制的诚实边界，每一级都由人工评审确认。"
                : "Customization depth unlocks in stages by committed scale — the honest boundary of small-batch customization, with human review confirming each stage."
            }
          />
          <ol className="mt-10 space-y-0 border-l border-gold/30 pl-6 sm:pl-8">
            {ladder.map((step) => (
              <li key={step.en} className="relative pb-8 last:pb-0">
                <span className="absolute -left-[31px] top-1 h-2.5 w-2.5 rounded-full border border-gold bg-ink sm:-left-[39px]" />
                <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                  <span className="font-display text-2xl text-gold">
                    {step.value}
                    <span className="ml-1 text-sm text-mist">{zh ? step.unitZh : step.unitEn}</span>
                  </span>
                  <span className="font-display text-base text-porcelain">{zh ? step.zh : step.en}</span>
                </div>
                <p className="mt-1 max-w-2xl text-sm leading-relaxed text-mist">{zh ? step.descZh : step.descEn}</p>
              </li>
            ))}
          </ol>
        </Section>
      </div>

      {/* Founder vs participant */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "权益对比" : "Rights, compared"}
          title={zh ? "发起人与参与者，各得其所" : "Founders and participants each keep their share"}
        />
        <div className="mt-10 grid gap-5 lg:grid-cols-2">
          <Card className="border-gold/25">
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-gold">
              {zh ? "发起人" : "Founder"}
            </p>
            <p className="font-display mt-2 text-xl text-porcelain">{zh ? "署名与优先权" : "Authorship and priority"}</p>
            <ul className="mt-4 space-y-2.5 text-sm leading-relaxed text-mist">
              <li className="flex items-start gap-2.5">
                <span className="mt-0.5 text-gold">◆</span>
                <span>{zh ? "创始版序列号与镌刻署名，写入每一瓶创始版。" : "Founder Edition serial and engraved name on every founder unit."}</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="mt-0.5 text-gold">▣</span>
                <span>{zh ? "专属 QR 档案页——项目故事以发起人的名字开篇。" : "An exclusive QR archive page — the project story opens with the founder's name."}</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="mt-0.5 text-gold">◈</span>
                <span>{zh ? "发起人数字印记与首铸预留。" : "A founder digital mark and first-pour reservation."}</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="mt-0.5 text-gold">❖</span>
                <span>{zh ? "门槛评审阶段的共创决策优先票。" : "Priority voice in co-creation decisions at each threshold review."}</span>
              </li>
            </ul>
            <p className="mt-4 text-xs text-mist">
              {zh ? "发起项目是核心序列会员权益；任何注册用户都可以参与。" : "Starting a project is a Core Sequence membership benefit; any registered user can participate."}
            </p>
          </Card>
          <Card>
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-mist">
              {zh ? "参与者" : "Participant"}
            </p>
            <p className="font-display mt-2 text-xl text-porcelain">{zh ? "份额与档案" : "A share and an archive"}</p>
            <ul className="mt-4 space-y-2.5 text-sm leading-relaxed text-mist">
              <li className="flex items-start gap-2.5">
                <span className="mt-0.5 text-gold">◇</span>
                <span>{zh ? "以 1–20 份的数量加入，预订即计入门槛进度。" : "Join with 1–20 units; reservations count toward every threshold."}</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="mt-0.5 text-gold">▤</span>
                <span>{zh ? "每一份都绑定共享批次档案与个人数字印记。" : "Each unit binds to the shared batch archive with a personal digital mark."}</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="mt-0.5 text-gold">✶</span>
                <span>{zh ? "标签与礼盒共创阶段的投票权。" : "A vote in the label and gift-box co-creation stages."}</span>
              </li>
              <li className="flex items-start gap-2.5">
                <span className="mt-0.5 text-gold">◉</span>
                <span>{zh ? "预订不收款：人工礼宾确认与合规审核之后才进入付款环节。" : "Reserving takes no payment: the payment step only begins after human concierge confirmation and compliance review."}</span>
              </li>
            </ul>
            <p className="mt-4 text-xs text-mist">
              {zh ? "交付后，你的份额出现在你的 Reserve 档案馆里。" : "After delivery, your share appears in your own Reserve archive."}
            </p>
          </Card>
        </div>
      </Section>

      {/* Projects */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section id="projects" className="scroll-mt-24 py-14 sm:py-20">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <SectionHeader
              eyebrow={zh ? "集结中" : "Gathering now"}
              title={zh ? "公开项目" : "Public projects"}
              description={
                zh
                  ? "已通过公开展示审核的项目。投票开放给所有人；加入需要注册账号。"
                  : "Projects cleared by public-display review. Voting is open to everyone; joining requires an account."
              }
            />
            <ButtonLink href="/co-create/new" variant="gold">
              {zh ? "发起一个项目" : "Start a project"}
            </ButtonLink>
          </div>
          <div className="mt-8">
            <CoCreateListClient zh={zh} projects={approved} />
          </div>

          {inReview.length > 0 && (
            <div className="mt-10">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-mist">
                {zh ? "评审中" : "In review"}
              </p>
              <div className="mt-3 space-y-3">
                {inReview.map((p) => (
                  <div
                    key={p.id}
                    className="flex flex-col gap-3 rounded-lg border border-hairline bg-veil/50 px-4 py-3.5 sm:flex-row sm:items-center sm:justify-between"
                  >
                    <div className="min-w-0">
                      <p className="font-display truncate text-sm text-porcelain">{p.title}</p>
                      <p className="mt-0.5 line-clamp-1 text-xs text-mist">{p.concept}</p>
                    </div>
                    <div className="flex shrink-0 items-center gap-2">
                      <Tag>{p.product_type}</Tag>
                      <StatusPill status={p.review_status} />
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-xs leading-relaxed text-mist">
                {zh
                  ? "评审中的项目已提交待审，通过公开展示审核后即开放投票与加入。"
                  : "Projects in review have been submitted for moderation; voting and joining open once public-display review passes."}
              </p>
            </div>
          )}
        </Section>
      </div>

      {/* Moderation & compliance */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "审核说明" : "Moderation"}
          title={zh ? "每个项目都经过人工评审" : "Every project passes human review"}
        />
        <div className="mt-6 flex flex-wrap gap-2">
          {[
            { en: "Sensitive content", zh: "敏感内容" },
            { en: "Alcohol compliance", zh: "酒类合规" },
            { en: "Minor safety", zh: "未成年人保护" },
            { en: "Copyright", zh: "版权" },
            { en: "Feasibility", zh: "可行性" },
            { en: "Public display", zh: "公开展示" },
            { en: "Trade eligibility", zh: "交易资格" },
          ].map((d) => (
            <Tag key={d.en} tone="gold">
              {zh ? d.zh : d.en}
            </Tag>
          ))}
        </div>
        <div className="mt-6 space-y-4">
          <Notice tone="ember" title={zh ? "审核维度" : "What review covers"}>
            {zh
              ? "共创项目在公开陈列与每一级门槛评审中，均接受敏感内容、酒类合规、未成年人保护、版权、可行性、公开展示与交易资格审查。涉及酒精的项目在实体交付前执行年龄与地区审核。"
              : "Co-creation projects are reviewed for sensitive content, alcohol compliance, minor safety, copyright, feasibility, public display, and trade eligibility — at public listing and at every threshold review. Alcohol-related projects run age and region checks before any physical delivery."}
          </Notice>
          <Notice tone="gold" title={zh ? "合规声明" : "Compliance"}>
            {pick(locale, complianceNotice)}
          </Notice>
        </div>
      </Section>
    </>
  );
}
