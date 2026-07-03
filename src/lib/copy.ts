import type { Locale } from "./types";

// Brand language — single source of truth for recurring copy.

export const brand = {
  en: "ZOTAIX is an AI concierge platform that turns emotions, relationships, scenarios, and budgets into bespoke spirits, fragrance directions, bottle design, gifting systems, and digital identity records.",
  zh: "ZOTAIX 卓序是一个 AI 礼宾式定制平台，用 AI 将人的情绪、关系、场景和预算，转化为可确认、可报价、可交付的酒饮、香氛、瓶身、包装与礼赠方案。",
};

export const headline = {
  en: "Turn emotions, relationships, and moments into collectible spirits, fragrances, bottles, and gifts.",
  zh: "把情绪、关系与时刻，铸造成可收藏的酒饮、香氛、瓶身与礼物。",
};

export const subheadline = {
  en: "ZOTAIX uses AI to understand your state, recipient, scenario, and budget, then generates liquid directions, fragrance directions, bottle visuals, label copy, gift-box stories, and digital identity records. Casual users can create emotional supply objects, while premium and enterprise clients can enter human concierge and bespoke gifting workflows.",
  zh: "ZOTAIX 用 AI 理解你的状态、赠予对象、场景与预算，生成酒体方向、香氛方向、瓶身视觉、瓶身文案、礼盒故事与数字身份档案。轻量用户可以创建情绪补给对象，高端与企业客户可以进入人工礼宾与定制礼赠流程。",
};

export const complianceNotice = {
  en: "AI-generated results are creative proposals. Real products require human confirmation, supply-chain confirmation, age and region compliance checks, and final quotation.",
  zh: "AI 生成结果为创意提案。实体产品需经人工确认、供应链确认、年龄与地区合规审核，以及最终报价。",
};

export const profileNotice = {
  en: "These details do not define who you are. They simply help ZOTAIX understand your preferred way of expression. You can skip, edit, or delete them at any time.",
  zh: "这些信息并不定义你是谁。它们只是帮助 ZOTAIX 理解你偏好的表达方式。你可以随时跳过、修改或删除。",
};

// "Order World" vocabulary mapping — used in membership / gamified areas only.
export const orderWorld: Record<string, { plain: string; order: string; zh: string }> = {
  user: { plain: "Account", order: "Order Builder", zh: "秩序构筑者" },
  profile: { plain: "Profile Center", order: "Order Hub", zh: "秩序中枢" },
  points: { plain: "Points", order: "Order Fragments", zh: "秩序残片" },
  quota: { plain: "Usage quota", order: "Order Energy", zh: "秩序能量" },
  membership: { plain: "Membership", order: "Core Sequence", zh: "核心序列" },
  checkin: { plain: "Check-in", order: "Anchor Order", zh: "锚定秩序" },
  redeem: { plain: "Redemption", order: "Physical Casting", zh: "实体铸造" },
  physical: { plain: "Physical spirit / fragrance", order: "Order Crystal", zh: "秩序结晶" },
  badge: { plain: "Digital badge", order: "Digital Mark", zh: "数字印记" },
  upgrade: { plain: "Membership upgrade", order: "Permission Leap", zh: "权限跃迁" },
  recharge: { plain: "Recharge", order: "Energy Injection", zh: "能量注入" },
};

export interface NavItem {
  href: string;
  en: string;
  zh: string;
  children?: NavItem[];
}

export const navigation: NavItem[] = [
  { href: "/concierge", en: "Concierge", zh: "AI 礼宾" },
  {
    href: "/forge",
    en: "Create",
    zh: "创造",
    children: [
      { href: "/forge", en: "Forge · AI Orchestration", zh: "Forge · AI 编排" },
      { href: "/studio", en: "Studio · Visual Preview", zh: "Studio · 视觉预览" },
      { href: "/design", en: "Design · Proposals & Versions", zh: "Design · 提案与版本" },
      { href: "/supply", en: "ZOTAIX Supply", zh: "情绪补给线" },
      { href: "/maison", en: "Maison ZOTAIX", zh: "高定礼赠线" },
    ],
  },
  {
    href: "/co-create",
    en: "Community",
    zh: "共创",
    children: [
      { href: "/co-create", en: "Co-Creation Pool", zh: "共创铸造池" },
      { href: "/market", en: "Creative Market", zh: "创意市场" },
      { href: "/trade", en: "Trade · Quotes & Rights", zh: "Trade · 报价与授权" },
      { href: "/cases", en: "Cases", zh: "案例" },
    ],
  },
  { href: "/reserve", en: "Reserve", zh: "档案馆" },
  { href: "/membership", en: "Core Sequence", zh: "核心序列" },
  {
    href: "/about",
    en: "More",
    zh: "更多",
    children: [
      { href: "/about", en: "About ZOTAIX", zh: "关于卓序" },
      { href: "/blog", en: "Journal", zh: "品牌志" },
      { href: "/download", en: "App Download", zh: "App 下载" },
      { href: "/wechat", en: "WeChat", zh: "微信公众号" },
      { href: "/social", en: "Global Social", zh: "全球社媒" },
    ],
  },
];

export const footerLegal: { href: string; en: string; zh: string }[] = [
  { href: "/legal/terms", en: "User Terms", zh: "用户协议" },
  { href: "/legal/privacy", en: "Privacy Policy", zh: "隐私政策" },
  { href: "/legal/cookies", en: "Cookie Policy", zh: "Cookie 政策" },
  { href: "/legal/ai", en: "AI Content Notice", zh: "AI 内容声明" },
  { href: "/legal/alcohol", en: "Alcohol Compliance", zh: "酒类合规" },
  { href: "/legal/minors", en: "Minor Protection", zh: "未成年人保护" },
  { href: "/legal/membership", en: "Membership Agreement", zh: "会员协议" },
  { href: "/legal/co-create", en: "Co-Creation Rules", zh: "共创规则" },
  { href: "/legal/trade", en: "Trade Market Rules", zh: "创意市场规则" },
  { href: "/legal/reserve", en: "Reserve Archive Rules", zh: "档案规则" },
  { href: "/legal/app", en: "App Privacy Notice", zh: "App 隐私声明" },
  { href: "/legal/contact", en: "Contact Us", zh: "联系我们" },
];

export function pick(locale: Locale, obj: { en: string; zh: string }): string {
  return locale === "zh" ? obj.zh : obj.en;
}
