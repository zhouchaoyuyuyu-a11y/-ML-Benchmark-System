import Link from "next/link";
import type { Metadata } from "next";
import { ButtonLink, Card, Meridian, Notice, PageHero, Section, SectionHeader, Tag } from "@/components/ui";
import QRCodeBox from "@/components/QRCodeBox";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = pageMetadata({
  title: "微信公众号 · ZOTAIX 卓序 WeChat Official Account",
  description:
    "关注 ZOTAIX 卓序官方公众号：微信内直达 AI 礼宾、共创池、会员权益与高定礼赠咨询。Follow the ZOTAIX 卓序 Official Account for the AI concierge, co-creation pool, membership benefits, and premium gifting inside WeChat.",
  path: "/wechat",
  keywords: ["ZOTAIX", "卓序", "微信公众号", "WeChat Official Account"],
});

export default async function WechatPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const wx = data.wechat_config;
  const settings = data.settings;

  const followerBenefits = [
    {
      icon: "◐",
      zhTitle: "AI 礼宾入口",
      enTitle: "AI concierge entry",
      desc: "在对话框里发一句今天的心情，礼宾用关键词与轻建议回应；礼物、香氛、酒体测试均可在微信内发起。",
      descEn: "Send one sentence about today and the concierge replies with keywords and a light suggestion — gift, fragrance, and spirit modes all start inside WeChat.",
    },
    {
      icon: "◆",
      zhTitle: "会员权益直达",
      enTitle: "Membership benefits",
      desc: "核心序列会员在公众号内查看配额、数字印记与专属通道，续费与升级提醒第一时间送达。",
      descEn: "Core Sequence members check quotas, digital marks, and member lanes in the account; renewal reminders arrive first here.",
    },
    {
      icon: "⬡",
      zhTitle: "共创活动通知",
      enTitle: "Co-creation events",
      desc: "共创池新项目、投票开启、达标进度与风味评审结果，按月推送共创月报。",
      descEn: "New pool projects, vote openings, threshold progress, and flavor-review results — plus a monthly co-creation digest.",
    },
    {
      icon: "◈",
      zhTitle: "高定礼赠咨询",
      enTitle: "Premium gifting inquiry",
      desc: "企业礼赠与私人高定可直接留言，人工礼宾一个工作日内回复，全程人工确认与报价。",
      descEn: "Leave a message for enterprise or private bespoke gifting; a human concierge replies within one business day with confirmation and quotation.",
    },
    {
      icon: "▤",
      zhTitle: "App 下载入口",
      enTitle: "App download",
      desc: "回复 APP 获取下载页链接，在系统浏览器中完成安装。",
      descEn: "Reply APP for the download page link and finish the install in your system browser.",
    },
  ];

  const shareCardPages = [
    { href: "/", label: "首页 · Home" },
    { href: "/concierge", label: "AI 礼宾 · Concierge" },
    { href: "/co-create", label: "共创池 · Co-Creation" },
    { href: "/reserve", label: "档案馆 · Reserve" },
    { href: "/supply", label: "情绪补给 · Supply" },
    { href: "/maison", label: "高定礼赠 · Maison" },
  ];

  return (
    <>
      <PageHero
        eyebrow="WeChat Official Account · 微信公众号"
        title={`${wx.official_account_name} 官方公众号`}
        description={
          "关注卓序，把 AI 礼宾放进微信：情绪对话、礼物灵感、共创池、会员权益与高定礼赠咨询，一个公众号全部直达。Follow ZOTAIX 卓序 to reach the AI concierge, co-creation pool, membership benefits, and premium gifting — all inside WeChat."
        }
      >
        <ButtonLink href="#follow" variant="gold">
          扫码关注 · Follow
        </ButtonLink>
        <ButtonLink href="/download" variant="outline">
          下载 App · Download
        </ButtonLink>
        <ButtonLink href="/concierge" variant="outline">
          网页版礼宾 · Web concierge
        </ButtonLink>
      </PageHero>

      {/* Follow + benefits */}
      <Section id="follow" className="py-14 sm:py-20">
        <div className="grid gap-10 lg:grid-cols-[minmax(0,320px)_1fr]">
          <div className="flex flex-col items-center gap-4 lg:items-start">
            <Card className="flex w-full flex-col items-center gap-3 text-center">
              {wx.qr_code_url ? (
                <img
                  src={wx.qr_code_url}
                  alt="ZOTAIX 卓序 微信公众号二维码"
                  width={180}
                  height={180}
                  className="rounded-lg border border-hairline bg-ink p-1"
                />
              ) : (
                <QRCodeBox seed="wechat:zotaix" size={180} label="微信扫码关注 · ZOTAIX 卓序" />
              )}
              <p className="text-sm text-porcelain">{wx.official_account_name}</p>
              <p className="text-xs leading-relaxed text-mist">
                打开微信 → 扫一扫 → 关注公众号
                <br />
                Open WeChat → Scan → Follow
              </p>
            </Card>
            {!wx.qr_code_url && (
              <Notice tone="gold" title="二维码由运营配置 · Operations-configured QR">
                公众号官方二维码图片由运营团队在管理后台配置；当前展示的是卓序数字身份码，配置完成后此处自动替换为官方二维码。The
                official QR image is set by the operations team in the admin console; this ZOTAIX identity code is shown
                until then and swaps automatically once configured.
              </Notice>
            )}
          </div>
          <div>
            <SectionHeader
              eyebrow="关注即获得 · What followers get"
              title="一个公众号，整条定制链"
              description="One account, the whole customization chain — 卓序公众号把平台的核心能力压缩进微信对话框。"
            />
            <div className="mt-6 grid gap-4 sm:grid-cols-2">
              {followerBenefits.map((b) => (
                <Card key={b.zhTitle} className="h-full">
                  <div className="flex items-start gap-3">
                    <span className="text-lg text-gold">{b.icon}</span>
                    <div>
                      <p className="font-display text-base text-porcelain">{b.zhTitle}</p>
                      <p className="text-xs uppercase tracking-wider text-mist">{b.enTitle}</p>
                      <p className="mt-2 text-sm leading-relaxed text-mist">{zh ? b.desc : b.descEn}</p>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </Section>

      {/* Menu structure — phone mockup */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <div className="grid items-start gap-10 lg:grid-cols-2">
            <div>
              <SectionHeader
                eyebrow="自定义菜单 · Custom menu"
                title="三栏十二格，按定制链排布"
                description="公众号底部菜单按「AI 礼宾 → 共创与会员 → 高定礼赠」的定制链排布，每一格都直达对应页面。The three-column menu mirrors the customization chain — every cell opens the matching page."
              />
              <div className="mt-6 space-y-3">
                {wx.menu_config.map((group) => (
                  <Card key={group.label} className="!p-4">
                    <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">{group.label}</p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {group.children.map((item) => (
                        <Link
                          key={item.label}
                          href={item.target}
                          className="rounded-full border border-hairline px-3 py-1 text-xs text-mist transition-colors hover:border-gold hover:text-gold"
                        >
                          {item.label} →
                        </Link>
                      ))}
                    </div>
                  </Card>
                ))}
              </div>
            </div>
            <div className="mx-auto w-full max-w-xs">
              <div className="rounded-[2rem] border border-hairline bg-obsidian p-3">
                <div className="overflow-hidden rounded-[1.4rem] border border-hairline bg-ink">
                  <div className="border-b border-hairline px-4 py-3 text-center">
                    <p className="text-sm text-porcelain">{wx.official_account_name}</p>
                    <p className="text-[10px] uppercase tracking-[0.2em] text-mist">Official Account</p>
                  </div>
                  <div className="grid grid-cols-3 gap-2 p-2.5">
                    {wx.menu_config.map((group) => (
                      <div key={group.label} className="flex flex-col justify-end gap-1.5">
                        {group.children.map((item) => (
                          <Link
                            key={item.label}
                            href={item.target}
                            className="rounded-md border border-hairline bg-veil px-1 py-1.5 text-center text-[10px] leading-tight text-porcelain transition-colors hover:border-gold hover:text-gold"
                          >
                            {item.label}
                          </Link>
                        ))}
                      </div>
                    ))}
                  </div>
                  <div className="grid grid-cols-3 border-t border-hairline">
                    {wx.menu_config.map((group) => (
                      <div
                        key={group.label}
                        className="border-r border-hairline px-1 py-2.5 text-center text-[10px] font-medium text-gold last:border-r-0"
                      >
                        ≡ {group.label}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <p className="mt-3 text-center text-xs text-mist">
                菜单结构示意 · 与后台 menu_config 实时一致
                <br />
                Menu preview, rendered live from the stored configuration
              </p>
            </div>
          </div>
        </Section>
      </div>

      {/* Auto-reply examples */}
      <Section className="py-14 sm:py-20">
        <SectionHeader
          eyebrow="自动回复 · Auto-reply"
          title="发一个关键词，礼宾接住你"
          description="关注后发送关键词，公众号即刻回复对应入口；以下为线上生效的自动回复配置。Send a keyword after following and the account replies instantly — these are the live auto-reply rules."
        />
        <div className="mx-auto mt-8 max-w-2xl space-y-6">
          {wx.auto_reply_config.map((rule) => (
            <div key={rule.trigger} className="space-y-2.5">
              {rule.trigger === "__follow__" ? (
                <p className="text-center text-xs tracking-wider text-mist">— 关注成功 · New follower —</p>
              ) : (
                <div className="flex justify-end">
                  <span className="max-w-[75%] rounded-2xl rounded-br-sm border border-gold/30 bg-gold/10 px-4 py-2 text-sm text-porcelain">
                    {rule.trigger}
                  </span>
                </div>
              )}
              <div className="flex items-start justify-start gap-2.5">
                <span className="font-display flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-hairline bg-veil text-xs text-gold">
                  卓
                </span>
                <span className="max-w-[80%] rounded-2xl rounded-bl-sm border border-hairline bg-veil px-4 py-2.5 text-sm leading-relaxed text-mist">
                  {rule.reply}
                </span>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {/* Customer service + membership */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-14 sm:py-20">
          <div className="grid gap-5 lg:grid-cols-2">
            <Card className="h-full">
              <div className="flex items-center gap-3">
                <p className="font-display text-xl text-porcelain">微信客服</p>
                <Tag tone="gold">WeChat customer service</Tag>
              </div>
              <p className="mt-3 text-sm leading-relaxed text-mist">
                公众号内发送「客服」即可转入人工礼宾，工作日 10:00–19:00（CST）在线；高定与企业礼赠留言一个工作日内回复。Send
                客服 in the account to reach a human concierge — business days 10:00–19:00 CST, with premium and
                enterprise inquiries answered within one business day.
              </p>
              <div className="mt-5 flex flex-wrap gap-3">
                {wx.customer_service_url ? (
                  <a
                    href={wx.customer_service_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center justify-center gap-2 rounded-md border border-gold bg-gold px-5 py-2.5 text-sm font-medium text-ink transition-colors hover:bg-gold-deep hover:text-porcelain"
                  >
                    打开微信客服 · Open WeChat service
                  </a>
                ) : (
                  <ButtonLink href="/legal/contact" variant="gold">
                    联系我们 · Contact us
                  </ButtonLink>
                )}
                <ButtonLink href="/maison#concierge" variant="outline">
                  人工礼宾 · Human concierge
                </ButtonLink>
              </div>
            </Card>
            <Card className="h-full">
              <div className="flex items-center gap-3">
                <p className="font-display text-xl text-porcelain">会员权益</p>
                <Tag tone="gold">Core Sequence</Tag>
              </div>
              <p className="mt-3 text-sm leading-relaxed text-mist">
                核心序列会员在公众号内直达专属权益：Lite 每日 {settings.lite_daily_chat} 次礼宾对话、每月{" "}
                {settings.lite_monthly_proposals} 份结构化提案；Pro 每日 {settings.pro_daily_chat} 次、每月{" "}
                {settings.pro_monthly_proposals} 份，并解锁导出与人工礼宾通道。Members reach their quotas, structured
                proposals, exports, and the concierge lane straight from WeChat.
              </p>
              <div className="mt-5 flex flex-wrap gap-3">
                <ButtonLink href="/membership" variant="gold">
                  查看核心序列 · Membership
                </ButtonLink>
                <ButtonLink href="/profile" variant="outline">
                  我的数字印记 · Digital marks
                </ButtonLink>
              </div>
            </Card>
          </div>
        </Section>
      </div>

      {/* Sharing + app download */}
      <Section className="py-14 sm:py-20">
        <div className="grid gap-10 lg:grid-cols-2">
          <div>
            <SectionHeader
              eyebrow="分享卡片 · Share cards"
              title="转发到微信，自动带卡片"
              description="平台各公开页面通过页面元数据输出分享标题、描述与封面图，链接贴进微信对话或朋友圈会自动展开为卡片。Share titles, descriptions, and covers are emitted via each page’s metadata, so links unfurl as cards in chats and Moments."
            />
            <div className="mt-6 flex flex-wrap gap-2">
              {shareCardPages.map((p) => (
                <Link
                  key={p.href}
                  href={p.href}
                  className="rounded-full border border-hairline px-3.5 py-1.5 text-xs text-mist transition-colors hover:border-gold hover:text-gold"
                >
                  {p.label}
                </Link>
              ))}
            </div>
            <p className="mt-4 text-xs leading-relaxed text-mist">
              公开的档案页与共创页同样附带卡片——被保存的对象，转发出去也是一件完整的作品。Public Reserve and
              co-creation pages carry cards too: a kept object travels as a finished piece.
            </p>
          </div>
          <Card className="flex h-full flex-col justify-between">
            <div>
              <p className="font-display text-xl text-porcelain">从公众号到 App</p>
              <p className="mt-3 text-sm leading-relaxed text-mist">
                公众号适合轻量对话与通知；完整的创作台、档案馆与共创池在 App 与网页应用中体验最佳。回复 APP
                获取链接，或直接前往下载页。The account handles light conversation and notices — the full studio,
                Reserve, and co-creation pool live in the app and web app.
              </p>
            </div>
            <div className="mt-5 flex flex-wrap gap-3">
              <ButtonLink href="/download" variant="gold">
                下载 ZOTAIX App · Download
              </ButtonLink>
              <ButtonLink href="/social" variant="outline">
                全球社媒矩阵 · Global social
              </ButtonLink>
            </div>
          </Card>
        </div>
        <Meridian className="mt-14" />
        <p className="mt-6 text-center text-xs leading-relaxed text-mist">
          {wx.official_account_name} · 公众号内容遵循平台合规准则：AI 生成结果为创意提案，实体产品需经人工确认与报价。
          <br />
          Account content follows platform compliance: AI results are creative proposals; physical products require
          human confirmation and quotation.
        </p>
      </Section>
    </>
  );
}
