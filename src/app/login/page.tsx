import type { Metadata } from "next";
import Link from "next/link";
import { Suspense } from "react";
import LoginClient from "./LoginClient";
import { Meridian, Section, Tag } from "@/components/ui";
import { brand, pick } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";

export const metadata: Metadata = {
  ...pageMetadata({
    title: "Sign in — return to your Order Hub",
    description:
      "Sign in to ZOTAIX to reopen your Order Hub: expression tags, preferences, recipient profiles, drafts, Reserve records, and Core Sequence benefits.",
    path: "/login",
  }),
  robots: { index: false, follow: true },
};

export default async function LoginPage() {
  const locale = await getLocale();
  const zh = locale === "zh";

  const unlocks = [
    {
      en: "Your Order Hub — expression tags, preferences, privacy controls",
      zh: "你的秩序中枢——表达标签、偏好与隐私控制",
    },
    {
      en: "Recipient profiles for the people you gift",
      zh: "为重要的人建立的赠予对象档案",
    },
    {
      en: "Daily Order Energy for the AI concierge",
      zh: "每日恢复的 AI 礼宾秩序能量",
    },
    {
      en: "Inspiration drafts and Reserve records with ZOTAIX IDs",
      zh: "灵感草稿与带 ZOTAIX ID 的档案记录",
    },
    {
      en: "Co-creation projects and your Core Sequence benefits",
      zh: "共创项目与你的核心序列权益",
    },
  ];

  return (
    <Section className="py-12 sm:py-20">
      <div className="grid gap-10 lg:grid-cols-2 lg:gap-16">
        {/* Brand panel */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.25em] text-gold">
            {zh ? "欢迎回来" : "Welcome back"}
          </p>
          <h1 className="font-display mt-3 text-3xl leading-tight text-porcelain sm:text-4xl">
            {zh ? "你的秩序中枢在等你" : "Your Order Hub is waiting"}
          </h1>
          <p className="mt-5 max-w-xl text-sm leading-relaxed text-mist sm:text-base">{pick(locale, brand)}</p>
          <Meridian className="my-8 max-w-xl" />
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-mist">
            {zh ? "登录后你可以继续" : "Signing in reopens"}
          </p>
          <ul className="mt-4 max-w-xl space-y-3">
            {unlocks.map((u) => (
              <li key={u.en} className="flex items-start gap-3 text-sm leading-relaxed text-mist">
                <span className="mt-0.5 text-gold">◈</span>
                {zh ? u.zh : u.en}
              </li>
            ))}
          </ul>
          <div className="mt-8 flex flex-wrap gap-2">
            <Tag tone="gold">{zh ? "秩序中枢 · Order Hub" : "Order Hub · 秩序中枢"}</Tag>
            <Tag tone="gold">{zh ? "核心序列 · Core Sequence" : "Core Sequence · 核心序列"}</Tag>
            <Tag tone="gold">{zh ? "数字印记 · Digital Mark" : "Digital Mark · 数字印记"}</Tag>
          </div>
          <p className="mt-8 text-sm text-mist">
            {zh ? "还没有账号？" : "No account yet?"}{" "}
            <Link href="/register" className="text-gold hover:underline">
              {zh ? "免费注册 →" : "Register free →"}
            </Link>
          </p>
        </div>

        {/* Form panel */}
        <div>
          <Suspense fallback={<div className="zx-skeleton h-96 rounded-lg" />}>
            <LoginClient zh={zh} />
          </Suspense>
        </div>
      </div>
    </Section>
  );
}
