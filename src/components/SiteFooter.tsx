import Link from "next/link";
import { brand, footerLegal, navigation } from "@/lib/copy";
import { db } from "@/lib/store";
import type { Locale } from "@/lib/types";

export default function SiteFooter({ locale }: { locale: Locale }) {
  const social = db()
    .social_accounts.filter((s) => s.enabled)
    .sort((a, b) => a.display_order - b.display_order);
  const zh = locale === "zh";

  return (
    <footer className="border-t border-hairline bg-obsidian/60">
      <div className="mx-auto grid w-full max-w-6xl gap-10 px-4 py-12 sm:px-6 lg:grid-cols-[2fr_1fr_1fr_1fr]">
        <div>
          <p className="font-display text-lg tracking-[0.18em] text-porcelain">
            ZOTAIX <span className="text-sm tracking-[0.3em] text-gold">卓序</span>
          </p>
          <p className="mt-3 max-w-md text-sm leading-relaxed text-mist">{zh ? brand.zh : brand.en}</p>
          <div className="mt-5 flex flex-wrap gap-2">
            {social.map((s) => (
              <a
                key={s.id}
                href={`${s.official_url}${s.tracking_params ? `?${s.tracking_params}` : ""}`}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-full border border-hairline px-3 py-1 text-xs text-mist transition-colors hover:border-gold hover:text-gold"
              >
                {s.platform}
              </a>
            ))}
            <Link
              href="/wechat"
              className="rounded-full border border-jade/40 px-3 py-1 text-xs text-jade transition-colors hover:bg-jade/10"
            >
              {zh ? "微信公众号" : "WeChat Official Account"}
            </Link>
          </div>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">{zh ? "平台" : "Platform"}</p>
          <ul className="mt-3 space-y-2 text-sm">
            {[
              { href: "/concierge", en: "AI Concierge", zh: "AI 礼宾" },
              { href: "/forge", en: "Forge", zh: "Forge 编排" },
              { href: "/studio", en: "Studio", zh: "Studio 预览" },
              { href: "/design", en: "Design Versions", zh: "Design 版本" },
              { href: "/trade", en: "Trade", zh: "Trade 报价" },
              { href: "/reserve", en: "Reserve Archive", zh: "Reserve 档案" },
            ].map((l) => (
              <li key={l.href}>
                <Link href={l.href} className="text-mist transition-colors hover:text-porcelain">
                  {zh ? l.zh : l.en}
                </Link>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">{zh ? "探索" : "Explore"}</p>
          <ul className="mt-3 space-y-2 text-sm">
            {[
              { href: "/supply", en: "ZOTAIX Supply", zh: "情绪补给" },
              { href: "/maison", en: "Maison ZOTAIX", zh: "高定礼赠" },
              { href: "/co-create", en: "Co-Creation Pool", zh: "共创池" },
              { href: "/market", en: "Creative Market", zh: "创意市场" },
              { href: "/membership", en: "Core Sequence", zh: "核心序列" },
              { href: "/download", en: "App Download", zh: "App 下载" },
              { href: "/social", en: "Global Social", zh: "全球社媒" },
            ].map((l) => (
              <li key={l.href}>
                <Link href={l.href} className="text-mist transition-colors hover:text-porcelain">
                  {zh ? l.zh : l.en}
                </Link>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">{zh ? "法律与合规" : "Legal"}</p>
          <ul className="mt-3 space-y-2 text-sm">
            {footerLegal.map((l) => (
              <li key={l.href}>
                <Link href={l.href} className="text-mist transition-colors hover:text-porcelain">
                  {zh ? l.zh : l.en}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <div className="border-t border-hairline">
        <div className="mx-auto flex w-full max-w-6xl flex-col gap-2 px-4 py-5 text-xs text-mist sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <p>© 2026 ZOTAIX. {zh ? "AI 生成内容为创意提案，实体交付需人工确认与合规审核。" : "AI-generated content is a creative proposal; physical delivery requires human confirmation and compliance review."}</p>
          <p>{zh ? "请理性饮酒 · 未成年人禁止饮酒" : "Drink responsibly · No alcohol for minors"}</p>
        </div>
      </div>
    </footer>
  );
}
