import Link from "next/link";
import type { Metadata } from "next";
import { Card } from "@/components/ui";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";
import ContactForm from "./ContactForm";

export const metadata: Metadata = pageMetadata({
  title: "Contact Us",
  description:
    "Reach the ZOTAIX team: human concierge email, enterprise inquiries, WeChat official account, global social channels, concierge hours, and a direct contact form.",
  path: "/legal/contact",
});

export default async function ContactPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const doc = db().legal_docs.find((d) => d.slug === "contact");

  const channels = [
    {
      icon: "◈",
      title: zh ? "人工礼宾" : "Human concierge",
      value: "concierge@zotaix.example",
      desc: zh
        ? "订单、定制、隐私、账户与举报的统一入口。"
        : "Orders, customization, privacy, accounts, and reports — one address for all of it.",
      href: "mailto:concierge@zotaix.example",
      external: true,
    },
    {
      icon: "◆",
      title: zh ? "企业与品牌合作" : "Enterprise & brand collaboration",
      value: "enterprise@zotaix.example",
      desc: zh
        ? "企业礼赠、品牌联名、城市伴手礼与批量项目。"
        : "Enterprise gifting, brand collaborations, city souvenirs, and volume programs.",
      href: "mailto:enterprise@zotaix.example",
      external: true,
    },
    {
      icon: "◉",
      title: zh ? "微信公众号" : "WeChat Official Account",
      value: "ZOTAIX 卓序",
      desc: zh
        ? "关注公众号，在微信内直达礼宾、共创与客服。"
        : "Follow the official account for concierge, co-creation, and support inside WeChat.",
      href: "/wechat",
      external: false,
    },
    {
      icon: "✳",
      title: zh ? "全球社媒" : "Global social",
      value: "Instagram · TikTok · X · YouTube · LinkedIn",
      desc: zh
        ? "官方账号矩阵——认准这些渠道，其他均非官方。"
        : "The official account matrix — these channels and no others are genuine.",
      href: "/social",
      external: false,
    },
  ];

  return (
    <div className="space-y-10">
      <header>
        <h1 className="font-display text-3xl text-porcelain sm:text-4xl">{zh ? "联系我们" : "Contact Us"}</h1>
        <p className="mt-3 text-xs uppercase tracking-[0.2em] text-gold">
          {zh ? "版本" : "Version"} {doc?.version ?? "1.0"} · {zh ? "生效日期" : "Effective date"}{" "}
          {doc?.effective_date ?? "2026-05-01"}
        </p>
        <p className="mt-4 text-sm leading-relaxed text-mist">
          {zh
            ? "每一条留言都由真人阅读。人工礼宾在工作日 10:00–19:00（CST）在线，一个工作日内回复。"
            : "Every message here is read by a human. The concierge is available on business days, 10:00–19:00 CST, and replies within one business day."}
        </p>
      </header>

      {/* Channel cards */}
      <div className="grid gap-4 sm:grid-cols-2">
        {channels.map((c) =>
          c.external ? (
            <a key={c.title} href={c.href} className="block">
              <Card hover className="h-full">
                <div className="flex items-start gap-3">
                  <span className="text-lg text-gold">{c.icon}</span>
                  <div>
                    <p className="font-display text-base text-porcelain">{c.title}</p>
                    <p className="mt-1 text-sm text-gold">{c.value}</p>
                    <p className="mt-1.5 text-xs leading-relaxed text-mist">{c.desc}</p>
                  </div>
                </div>
              </Card>
            </a>
          ) : (
            <Link key={c.title} href={c.href} className="block">
              <Card hover className="h-full">
                <div className="flex items-start gap-3">
                  <span className="text-lg text-gold">{c.icon}</span>
                  <div>
                    <p className="font-display text-base text-porcelain">{c.title}</p>
                    <p className="mt-1 text-sm text-gold">{c.value}</p>
                    <p className="mt-1.5 text-xs leading-relaxed text-mist">{c.desc}</p>
                  </div>
                </div>
              </Card>
            </Link>
          ),
        )}
      </div>

      {/* Hours + registered address */}
      <div className="grid gap-4 sm:grid-cols-2">
        <Card>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
            {zh ? "礼宾时间" : "Concierge hours"}
          </p>
          <p className="font-display mt-2 text-lg text-porcelain">10:00 – 19:00 CST</p>
          <p className="mt-1.5 text-sm leading-relaxed text-mist">
            {zh
              ? "工作日在线。非工作时间的留言会在下一个工作日优先处理。"
              : "Business days. Messages outside these hours are answered first thing the next business day."}
          </p>
        </Card>
        <Card>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
            {zh ? "注册地址与运营者信息" : "Registered address & operator information"}
          </p>
          <p className="mt-2 text-sm leading-relaxed text-mist">
            {zh
              ? "运营者的法定名称、注册地址与邮寄地址，公示于各应用商店与微信公众号的运营者信息栏。"
              : "The operator's legal name, registered address, and postal address are published in the operator information section of the app stores and the WeChat official account."}
          </p>
        </Card>
      </div>

      {/* Contact form */}
      <Card>
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
          {zh ? "直接留言" : "Write to us directly"}
        </p>
        <p className="font-display mt-2 text-lg text-porcelain">
          {zh ? "一条留言，直达礼宾队列" : "One message, straight into the concierge queue"}
        </p>
        <p className="mt-1.5 text-sm leading-relaxed text-mist">
          {zh
            ? "无需登录。留下联系方式与想说的话，人工礼宾会回复你。"
            : "No sign-in needed. Leave a contact and your message; a human concierge will get back to you."}
        </p>
        <div className="mt-5">
          <ContactForm zh={zh} />
        </div>
      </Card>
    </div>
  );
}
