import Link from "next/link";
import type { Metadata } from "next";
import JsonLd from "@/components/JsonLd";
import { ButtonLink, Card, Meridian, Notice, Section, SectionHeader, Tag } from "@/components/ui";
import { siteUrl } from "@/lib/config";
import { brand, complianceNotice, pick } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "About ZOTAIX — the AI concierge customization platform",
  description:
    "ZOTAIX is a product, a service, and a repeat-purchase system: an AI concierge that turns emotions, relationships, scenarios, and budgets into bespoke spirits, fragrance directions, bottle design, gifting systems, and digital identity records.",
  path: "/about",
});

export default async function AboutPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const socialNames = data.social_accounts
    .filter((s) => s.enabled)
    .sort((a, b) => a.display_order - b.display_order)
    .map((s) => s.platform);
  const wechatName = data.wechat_config.official_account_name;

  const notCards = [
    {
      en: "Not a liquor website",
      zhT: "不是一个卖酒网站",
      descEn:
        "There is no catalog on the front door and no cart behind it. ZOTAIX opens with a question about your state, and every physical bottle passes human confirmation, age and region checks, and a final quotation before it exists.",
      descZh:
        "首页没有商品目录，背后也没有购物车。ZOTAIX 以一个关于你状态的问题开场；每一瓶实体酒都要经过人工确认、年龄与地区审核和最终报价才会诞生。",
    },
    {
      en: "Not just a chatbot",
      zhT: "不只是一个聊天机器人",
      descEn:
        "Conversation is only the entrance. The output is a structured object — liquid direction, fragrance direction, names, label copy, digital mark — that can be saved, versioned, shared, co-created, quoted, and archived.",
      descZh:
        "对话只是入口。产出是一个结构化对象——酒体方向、香氛方向、命名、瓶身文案、数字印记——可以被保存、迭代、分享、共创、报价与归档。",
    },
    {
      en: "Not just a luxury gift brand",
      zhT: "不只是一个奢侈礼品品牌",
      descEn:
        "Gifting is one outcome among many. The same chain serves a student's zero-proof exam ration, a member's private anniversary fragrance, and a hotel group's 300-unit appreciation program.",
      descZh:
        "礼赠只是众多结果之一。同一条定制链既服务学生的零酒精考试补给，也服务会员的私人纪念日香氛，以及酒店集团 300 份的答谢项目。",
    },
  ];

  const chain = [
    {
      href: "/forge",
      name: "Forge",
      en: "AI orchestration: one emotion becomes a liquid direction, a fragrance direction, names, and label copy.",
      zhD: "AI 编排：一种情绪生成酒体方向、香氛方向、命名与瓶身文案。",
    },
    {
      href: "/studio",
      name: "Studio",
      en: "Visual preview: bottles, labels, packaging, and emotional cards rendered before anything is produced.",
      zhD: "视觉预览：在任何生产发生之前，预览瓶身、标签、包装与情绪卡片。",
    },
    {
      href: "/design",
      name: "Design",
      en: "Proposals and versions: every iteration is named, hashed, and kept, so nothing you liked is ever lost.",
      zhD: "提案与版本：每次迭代都被命名、哈希并保留，你喜欢过的从不丢失。",
    },
    {
      href: "/trade",
      name: "Trade",
      en: "Quotes and rights: human-reviewed quotations, authorizations, and enterprise sample paths.",
      zhD: "报价与授权：人工评审的报价、授权与企业打样路径。",
    },
    {
      href: "/reserve",
      name: "Reserve",
      en: "Identity records: a ZOTAIX ID, QR-bound certificate, aftercare, and replenishment entry for every object.",
      zhD: "身份档案：每个对象都有 ZOTAIX ID、QR 绑定证书、售后与补铸入口。",
    },
  ];

  const principles = [
    {
      en: "Understanding precedes commerce",
      zhT: "理解先于交易",
      descEn:
        "The concierge collects emotion, recipient, scenario, and budget before anything is proposed. A price appears only after a person has been understood.",
      descZh: "礼宾先收集情绪、对象、场景与预算，然后才有任何提案。价格只在一个人被理解之后出现。",
    },
    {
      en: "Objects before carts",
      zhT: "先有对象，后有订单",
      descEn:
        "Users create and save a personalized object first, then decide whether it becomes physical. Nothing on this platform pushes you toward checkout.",
      descZh: "用户先创造并保存一个属于自己的对象，之后再决定它是否成为实物。平台上没有任何东西把你推向结算页。",
    },
    {
      en: "Human confirmation for physical delivery",
      zhT: "实体交付必经人工确认",
      descEn:
        "Every casting, co-creation run, and enterprise program is confirmed by a human concierge and the supply chain before production begins.",
      descZh: "每一次实体铸造、共创批次与企业项目，都在生产开始前由人工礼宾与供应链确认。",
    },
    {
      en: "Archives outlast bottles",
      zhT: "档案比瓶子活得更久",
      descEn:
        "A bottle empties and a fragrance fades; the Reserve record of why they existed — with its QR certificate — stays, and replenishment attaches to the record.",
      descZh: "酒会喝完，香会散去；记录它们为何存在的 Reserve 档案与 QR 证书会留下，补铸挂在档案上。",
    },
    {
      en: "Compliance by design",
      zhT: "合规内建于设计",
      descEn:
        "Age gates, region checks, minor protection, and multi-dimension moderation are built into the flows, not appended to them.",
      descZh: "年龄门、地区审核、未成年人保护与多维度审核内建于流程之中，而非事后补丁。",
    },
  ];

  const moderationDimensions = zh
    ? ["敏感内容", "酒类合规", "未成年人安全", "版权与原创", "生产可行性", "公开展示", "交易资格", "医疗声称", "虚假承诺", "站外交易"]
    : [
        "Sensitive content",
        "Alcohol compliance",
        "Minor safety",
        "Copyright & originality",
        "Production feasibility",
        "Public display",
        "Trade eligibility",
        "Medical claims",
        "False promises",
        "External transactions",
      ];

  const aboutJsonLd = {
    "@context": "https://schema.org",
    "@type": "AboutPage",
    name: "About ZOTAIX",
    url: `${siteUrl}/about`,
    description: brand.en,
    inLanguage: ["en", "zh-CN"],
    mainEntity: {
      "@type": "Organization",
      name: "ZOTAIX",
      alternateName: "ZOTAIX 卓序",
      url: siteUrl,
      email: "concierge@zotaix.example",
    },
  };

  return (
    <>
      <JsonLd data={aboutJsonLd} />

      {/* Hero: both brand sentences */}
      <div className="zx-grid-bg border-b border-hairline">
        <Section className="py-16 sm:py-24">
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.25em] text-gold">
            {zh ? "关于卓序" : "About ZOTAIX"}
          </p>
          <h1 className="font-display max-w-4xl text-2xl leading-snug text-porcelain sm:text-4xl sm:leading-tight">
            {brand.en}
          </h1>
          <p className="mt-6 max-w-3xl text-sm leading-relaxed text-mist sm:text-lg">{brand.zh}</p>
          <div className="mt-9 flex flex-wrap gap-3">
            <ButtonLink href="/concierge" variant="gold">
              {zh ? "启动 AI 礼宾" : "Start the AI concierge"}
            </ButtonLink>
            <ButtonLink href="/cases" variant="outline">
              {zh ? "查看客户案例" : "See client cases"}
            </ButtonLink>
          </div>
        </Section>
      </div>

      {/* Platform definition + what we are not */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "平台定义" : "What ZOTAIX is"}
          title={zh ? "AI 礼宾式定制平台：产品 + 服务 + 复购体系" : "An AI concierge customization platform: product + service + repeat purchase"}
          description={
            zh
              ? "产品，是被生成并保存的对象——酒饮、香氛、瓶身、礼盒与数字档案。服务，是从 AI 理解到人工礼宾确认的全程陪伴。复购，是档案驱动的补铸、售后与年复一年的仪式。三者构成一个闭环，而不是一次交易。"
              : "The product is the generated, saved object — spirits, fragrances, bottles, gift boxes, and digital records. The service is the accompanied journey from AI understanding to human concierge confirmation. The repeat purchase is archive-driven replenishment, aftercare, and rituals that return year after year. Together they form a loop, not a transaction."
          }
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {notCards.map((c) => (
            <Card key={c.en} className="h-full">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-ember/80">
                {zh ? "我们不是" : "What we are not"}
              </p>
              <p className="font-display mt-2 text-lg text-porcelain">{zh ? c.zhT : c.en}</p>
              <p className="mt-2.5 text-sm leading-relaxed text-mist">{zh ? c.descZh : c.descEn}</p>
            </Card>
          ))}
        </div>
      </Section>

      {/* Five-module object chain */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "对象链" : "The object chain"}
            title={zh ? "五个模块，一条对象的生命线" : "Five modules, one lifeline for every object"}
            description={
              zh
                ? "每个对象都沿同一条链流动：被编排、被预览、被版本化、被报价、被归档。"
                : "Every object moves along the same chain: orchestrated, previewed, versioned, quoted, archived."
            }
          />
          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {chain.map((m, i) => (
              <Link key={m.name} href={m.href} className="block h-full">
                <Card hover className="h-full">
                  <p className="font-display text-2xl text-gold/60">{String(i + 1).padStart(2, "0")}</p>
                  <p className="font-display mt-2 text-base text-porcelain">{m.name}</p>
                  <p className="mt-2 text-xs leading-relaxed text-mist">{zh ? m.zhD : m.en}</p>
                  <p className="mt-3 text-xs text-gold">{zh ? "进入" : "Enter"} →</p>
                </Card>
              </Link>
            ))}
          </div>
        </Section>
      </div>

      {/* Dual product lines */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "双产品线" : "Dual product lines"}
          title={zh ? "同一条链，两种交付深度" : "One chain, two depths of delivery"}
        />
        <div className="mt-10 grid gap-5 lg:grid-cols-2">
          <Card className="border-gold/25">
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-gold">Maison ZOTAIX</p>
            <p className="font-display mt-2 text-xl text-porcelain">{zh ? "高定礼赠线" : "The premium line"}</p>
            <p className="mt-3 text-sm leading-relaxed text-mist">
              {zh
                ? "企业礼赠、客户答谢、宴席与私人庆典、品牌联名与城市伴手礼。AI 礼宾收集场景与预算，人工礼宾确认每一步：设计、打样、报价、交付、售后，以及绑定 Reserve 的档案身份。"
                : "Enterprise gifting, client appreciation, banquets and private celebrations, brand collaborations and city souvenirs. The AI concierge collects scenario and budget; a human concierge confirms every step — design, sampling, quotation, delivery, aftercare, and a Reserve-bound identity."}
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              <Tag tone="gold">{zh ? "人工礼宾" : "Human concierge"}</Tag>
              <Tag tone="gold">{zh ? "报价与打样" : "Quotes & samples"}</Tag>
              <Tag tone="gold">{zh ? "企业项目" : "Enterprise programs"}</Tag>
            </div>
            <div className="mt-5">
              <ButtonLink href="/maison" variant="gold">{zh ? "进入 Maison" : "Enter Maison"}</ButtonLink>
            </div>
          </Card>
          <Card className="border-supply/25">
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-supply">ZOTAIX Supply</p>
            <p className="font-display mt-2 text-xl text-porcelain">{zh ? "情绪补给线" : "The emotional supply line"}</p>
            <p className="mt-3 text-sm leading-relaxed text-mist">
              {zh
                ? "生日、失恋恢复、考试解压、职场情绪、友情与派对。轻量、俏皮、可分享：情绪命名、个性标签、数字徽章、低酒精与零酒精选项，以及十人即可开启的低门槛共创。"
                : "Birthdays, breakup recovery, exam stress, workplace feelings, friendship, and parties. Light, playful, shareable: emotional naming, personalized labels, digital badges, low- and zero-proof options, and low-barrier co-creation that opens at ten people."}
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              <Tag tone="supply">{zh ? "情绪卡片" : "Emotional cards"}</Tag>
              <Tag tone="supply">{zh ? "零酒精友好" : "Zero-proof friendly"}</Tag>
              <Tag tone="supply">{zh ? "低门槛共创" : "Low-barrier co-creation"}</Tag>
            </div>
            <div className="mt-5">
              <ButtonLink href="/supply" variant="supply">{zh ? "进入 Supply" : "Enter Supply"}</ButtonLink>
            </div>
          </Card>
        </div>
      </Section>

      {/* Principles */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "原则" : "Principles"}
            title={zh ? "五条不会让步的原则" : "Five principles we do not trade away"}
          />
          <div className="mt-10 space-y-0">
            {principles.map((p, i) => (
              <div
                key={p.en}
                className="grid gap-2 border-b border-hairline py-6 last:border-0 sm:grid-cols-[80px_260px_1fr] sm:gap-6"
              >
                <p className="font-display text-2xl text-gold/50">{String(i + 1).padStart(2, "0")}</p>
                <p className="font-display text-base text-porcelain">{zh ? p.zhT : p.en}</p>
                <p className="text-sm leading-relaxed text-mist">{zh ? p.descZh : p.descEn}</p>
              </div>
            ))}
          </div>
        </Section>
      </div>

      {/* Compliance & safety */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow={zh ? "合规与安全" : "Compliance & safety"}
          title={zh ? "创意自由，交付审慎" : "Free in creation, careful in delivery"}
        />
        <div className="mt-8">
          <Notice tone="gold" title={zh ? "平台合规声明" : "Platform compliance statement"}>
            {pick(locale, complianceNotice)}
          </Notice>
        </div>
        <div className="mt-6 grid gap-4 lg:grid-cols-3">
          <Card className="h-full">
            <p className="font-display text-base text-porcelain">{zh ? "年龄与地区审核" : "Age & region checks"}</p>
            <p className="mt-2 text-sm leading-relaxed text-mist">
              {zh
                ? "涉酒页面设有年龄门；任何涉及酒类实体交付的流程都在报价前完成年龄与地区合规审核。不满足条件的请求会被礼宾引导至零酒精或纯数字路径。"
                : "Alcohol-related pages carry an age gate, and every flow that could deliver physical alcohol completes age and region compliance checks before quotation. Requests that do not qualify are guided by the concierge toward zero-proof or digital-only paths."}
            </p>
          </Card>
          <Card className="h-full">
            <p className="font-display text-base text-porcelain">{zh ? "未成年人保护" : "Minor protection"}</p>
            <p className="mt-2 text-sm leading-relaxed text-mist">
              {zh
                ? "面向学生与年轻社群的补给线以零酒精产品为默认；任何面向未成年人的酒类表达都会被审核拦截。情绪表达不需要酒精也能成立。"
                : "Supply-line projects for students and young communities default to zero-proof products, and any alcohol expression aimed at minors is intercepted in review. Emotional expression stands on its own without alcohol."}
            </p>
            <div className="mt-3">
              <Link href="/legal/minors" className="text-xs text-gold hover:underline">
                {zh ? "未成年人保护声明 →" : "Minor Protection Notice →"}
              </Link>
            </div>
          </Card>
          <Card className="h-full">
            <p className="font-display text-base text-porcelain">{zh ? "十个审核维度" : "Ten moderation dimensions"}</p>
            <p className="mt-2 text-sm leading-relaxed text-mist">
              {zh
                ? "所有公开内容、共创项目与交易请求都会经过多维度人工审核："
                : "Every public item, co-creation project, and trade request passes human review across these dimensions:"}
            </p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              {moderationDimensions.map((d) => (
                <Tag key={d}>{d}</Tag>
              ))}
            </div>
          </Card>
        </div>
      </Section>

      {/* Team / atelier */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <SectionHeader
            eyebrow={zh ? "团队与工坊" : "The people behind it"}
            title={zh ? "礼宾团队、设计工坊与供应链伙伴" : "A concierge team, a design atelier, supply-chain partners"}
          />
          <div className="mt-10 grid gap-6 lg:grid-cols-3">
            <div>
              <p className="font-display text-base text-porcelain">{zh ? "人工礼宾团队" : "The concierge team"}</p>
              <p className="mt-2 text-sm leading-relaxed text-mist">
                {zh
                  ? "每一个走向实体的对象都由一位人工礼宾接手：核对场景、确认预算、协调打样与交期，并在工作日 10:00–19:00（CST）内回复。AI 负责理解与提案，礼宾负责承诺与兑现。"
                  : "Every object headed for the physical world is taken over by a human concierge who verifies the scenario, confirms the budget, coordinates samples and timelines, and replies within business hours, 10:00–19:00 CST. The AI understands and proposes; the concierge promises and delivers."}
              </p>
            </div>
            <div>
              <p className="font-display text-base text-porcelain">{zh ? "设计工坊" : "The design atelier"}</p>
              <p className="mt-2 text-sm leading-relaxed text-mist">
                {zh
                  ? "工坊将 AI 提案打磨为可生产的设计：瓶型选择、标签工艺、礼盒结构与雕刻细节。每个版本都被哈希归档，客户确认的那一版就是进入生产的那一版。"
                  : "The atelier refines AI proposals into producible designs — bottle selection, label finishing, gift-box construction, engraving details. Every version is hashed and archived; the version the client confirms is the version that enters production."}
              </p>
            </div>
            <div>
              <p className="font-display text-base text-porcelain">{zh ? "供应链伙伴" : "Supply-chain partners"}</p>
              <p className="mt-2 text-sm leading-relaxed text-mist">
                {zh
                  ? "合作酒厂、调香实验室与包装工厂在标准化基酒与基础香型之上完成个性化表达。定制深度随批量分级解锁——50 瓶开放标签与礼盒主题，100 瓶开放风味方向评审，300 瓶进入企业礼赠评审。"
                  : "Partner distilleries, fragrance labs, and packaging workshops carry personalized expression on standardized base liquids and accords. Depth unlocks in stages with volume — 50 bottles open label and box theming, 100 bottles open flavor-direction review, 300 bottles enter enterprise gifting review."}
              </p>
            </div>
          </div>
        </Section>
      </div>

      {/* Contact strip */}
      <Section className="py-12 sm:py-16">
        <Meridian className="mb-10" />
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <p className="text-xs uppercase tracking-wider text-mist">{zh ? "写信给礼宾" : "Write to the concierge"}</p>
            <a href="mailto:concierge@zotaix.example" className="font-display mt-1 block text-sm text-porcelain transition-colors hover:text-gold">
              concierge@zotaix.example
            </a>
            <p className="mt-1 text-xs text-mist">{zh ? "工作日 10:00–19:00（CST）内回复" : "Replies on business days, 10:00–19:00 CST"}</p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wider text-mist">{zh ? "联系与服务" : "Contact & service"}</p>
            <Link href="/legal/contact" className="font-display mt-1 block text-sm text-porcelain transition-colors hover:text-gold">
              {zh ? "联系我们" : "Contact us"}
            </Link>
            <p className="mt-1 text-xs text-mist">{zh ? "服务时间、地址与法务信息" : "Service hours, addresses, legal details"}</p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wider text-mist">{zh ? "微信公众号" : "WeChat Official Account"}</p>
            <Link href="/wechat" className="font-display mt-1 block text-sm text-porcelain transition-colors hover:text-gold">
              {wechatName}
            </Link>
            <p className="mt-1 text-xs text-mist">{zh ? "微信内直达礼宾与共创" : "Concierge and co-creation inside WeChat"}</p>
          </div>
          <div>
            <p className="text-xs uppercase tracking-wider text-mist">{zh ? "全球社媒" : "Global social"}</p>
            <Link href="/social" className="font-display mt-1 block text-sm text-porcelain transition-colors hover:text-gold">
              {zh ? "官方账号矩阵" : "Official accounts"}
            </Link>
            <p className="mt-1 text-xs text-mist">{socialNames.slice(0, 5).join(" · ")}</p>
          </div>
        </div>
      </Section>
    </>
  );
}
