import type { Metadata } from "next";
import Link from "next/link";
import { Suspense } from "react";
import RegisterClient from "./RegisterClient";
import { Meridian, Section, Tag } from "@/components/ui";
import { brand, pick } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";

export const metadata: Metadata = {
  ...pageMetadata({
    title: "Register — open your Order Hub",
    description:
      "Create a free ZOTAIX account to save your profile, recipient profiles, and inspiration drafts, receive daily Order Energy, keep basic Reserve records, and join co-creation projects.",
    path: "/register",
  }),
  robots: { index: false, follow: true },
};

export default async function RegisterPage() {
  const locale = await getLocale();
  const zh = locale === "zh";

  const registered = [
    {
      en: "Save your profile, recipient profiles, and inspiration drafts",
      zh: "保存你的档案、赠予对象档案与灵感草稿",
    },
    { en: "Free daily Order Energy for the AI concierge", zh: "每日免费恢复的 AI 礼宾秩序能量" },
    { en: "Basic Reserve records with ZOTAIX IDs", zh: "带 ZOTAIX ID 的基础档案记录" },
    { en: "Join co-creation projects and vote", zh: "加入共创项目并参与投票" },
  ];

  const guest = [
    { en: "1–3 lightweight concierge replies", zh: "1–3 次轻量礼宾回应" },
    { en: "One temporary test", zh: "一次临时体验测试" },
    { en: "One basic emotional card", zh: "一张基础情绪卡片" },
    { en: "Browse public cases and archives", zh: "浏览公开案例与档案" },
  ];

  return (
    <Section className="py-12 sm:py-20">
      <div className="grid gap-10 lg:grid-cols-2 lg:gap-16">
        {/* Brand panel */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.25em] text-gold">
            {zh ? "成为秩序构筑者" : "Become an Order Builder"}
          </p>
          <h1 className="font-display mt-3 text-3xl leading-tight text-porcelain sm:text-4xl">
            {zh ? "先创造对象，再决定它成为什么" : "Create the object first — decide what it becomes later"}
          </h1>
          <p className="mt-5 max-w-xl text-sm leading-relaxed text-mist sm:text-base">{pick(locale, brand)}</p>
          <Meridian className="my-8 max-w-xl" />

          <div className="grid max-w-xl gap-4 sm:grid-cols-2">
            <div className="rounded-lg border border-gold/30 bg-gold/5 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
                {zh ? "注册解锁（免费）" : "Registration unlocks (free)"}
              </p>
              <ul className="mt-3 space-y-2.5">
                {registered.map((u) => (
                  <li key={u.en} className="flex items-start gap-2 text-sm leading-relaxed text-mist">
                    <span className="mt-0.5 text-gold">✓</span>
                    {zh ? u.zh : u.en}
                  </li>
                ))}
              </ul>
            </div>
            <div className="rounded-lg border border-hairline bg-obsidian/60 p-4">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-mist">
                {zh ? "访客可以" : "As a guest"}
              </p>
              <ul className="mt-3 space-y-2.5">
                {guest.map((u) => (
                  <li key={u.en} className="flex items-start gap-2 text-sm leading-relaxed text-mist">
                    <span className="mt-0.5 text-mist">·</span>
                    {zh ? u.zh : u.en}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="mt-8 flex flex-wrap gap-2">
            <Tag tone="gold">{zh ? "秩序中枢 · Order Hub" : "Order Hub · 秩序中枢"}</Tag>
            <Tag tone="gold">{zh ? "秩序能量 · Order Energy" : "Order Energy · 秩序能量"}</Tag>
            <Tag tone="gold">{zh ? "数字印记 · Digital Mark" : "Digital Mark · 数字印记"}</Tag>
          </div>
          <p className="mt-8 text-sm text-mist">
            {zh ? "已经有账号？" : "Already have an account?"}{" "}
            <Link href="/login" className="text-gold hover:underline">
              {zh ? "直接登录 →" : "Sign in instead →"}
            </Link>
          </p>
        </div>

        {/* Form panel */}
        <div>
          <Suspense fallback={<div className="zx-skeleton h-96 rounded-lg" />}>
            <RegisterClient zh={zh} />
          </Suspense>
        </div>
      </div>
    </Section>
  );
}
