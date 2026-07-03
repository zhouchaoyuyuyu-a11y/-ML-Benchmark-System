"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState, type FormEvent } from "react";
import { Button, Notice } from "@/components/ui";

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

const DEMO_PASSWORD = "zotaix-demo";

const DEMO_ACCOUNTS: { email: string; en: string; zh: string }[] = [
  { email: "member@zotaix.demo", en: "Core Sequence Pro member", zh: "核心序列 Pro 会员" },
  { email: "lite@zotaix.demo", en: "Core Sequence Lite member", zh: "核心序列 Lite 会员" },
  { email: "user@zotaix.demo", en: "Free registered account", zh: "免费注册账号" },
  { email: "admin@zotaix.demo", en: "Operations admin", zh: "运营管理员" },
];

export default function LoginClient({ zh = false }: { zh?: boolean }) {
  const router = useRouter();
  const params = useSearchParams();
  const next = params.get("next");

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = (await res.json().catch(() => ({}))) as { ok?: boolean; error?: string };
      if (!res.ok || !data.ok) {
        setError(data.error ?? (zh ? "登录没有成功，请检查邮箱与密码。" : "Sign-in did not go through — check your email and password."));
        setLoading(false);
        return;
      }
      const target = next && next.startsWith("/") ? next : "/profile";
      router.push(target);
      router.refresh();
    } catch {
      setError(zh ? "网络波动了一下，请再试一次。" : "The connection wobbled — please try again.");
      setLoading(false);
    }
  }

  function fillDemo(demoEmail: string) {
    setEmail(demoEmail);
    setPassword(DEMO_PASSWORD);
    setError(null);
  }

  return (
    <div className="space-y-5">
      <form onSubmit={submit} className="zx-card p-6 sm:p-8">
        <h2 className="font-display text-xl text-porcelain">{zh ? "登录 ZOTAIX" : "Sign in to ZOTAIX"}</h2>
        <p className="mt-1.5 text-sm text-mist">
          {next
            ? zh
              ? "登录后会带你回到刚才的页面。"
              : "You will return to where you left off after signing in."
            : zh
              ? "登录后直接进入你的秩序中枢。"
              : "You land in your Order Hub right after signing in."}
        </p>

        <div className="mt-6 space-y-4">
          <label className="block">
            <span className="text-xs uppercase tracking-wider text-mist">{zh ? "邮箱" : "Email"}</span>
            <input
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder={zh ? "you@example.com" : "you@example.com"}
              className={`mt-1.5 ${inputCls}`}
            />
          </label>
          <label className="block">
            <span className="text-xs uppercase tracking-wider text-mist">{zh ? "密码" : "Password"}</span>
            <input
              type="password"
              required
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className={`mt-1.5 ${inputCls}`}
            />
          </label>
        </div>

        {error && (
          <div className="mt-4">
            <Notice tone="ember">{error}</Notice>
          </div>
        )}

        <Button type="submit" variant="gold" className="mt-6 w-full" disabled={loading}>
          {loading ? (zh ? "登录中…" : "Signing in…") : zh ? "登录" : "Sign in"}
        </Button>

        <p className="mt-4 text-center text-sm text-mist">
          {zh ? "还没有账号？" : "New here?"}{" "}
          <Link
            href={next ? `/register?next=${encodeURIComponent(next)}` : "/register"}
            className="text-gold hover:underline"
          >
            {zh ? "免费注册 →" : "Register free →"}
          </Link>
        </p>
      </form>

      {/* Demo accounts */}
      <div className="rounded-lg border border-hairline bg-obsidian/60 p-5">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
          {zh ? "演示账号 · 一键填入" : "Demo accounts · tap to fill"}
        </p>
        <p className="mt-2 text-xs leading-relaxed text-mist">
          {zh
            ? `以下账号预置了示例档案、草稿与共创数据，便于体验各个序列等级。统一密码：${DEMO_PASSWORD}`
            : `These accounts come pre-seeded with sample archives, drafts, and co-creation data so you can explore every sequence level. Shared password: ${DEMO_PASSWORD}`}
        </p>
        <div className="mt-3 grid gap-2 sm:grid-cols-2">
          {DEMO_ACCOUNTS.map((d) => (
            <button
              key={d.email}
              type="button"
              onClick={() => fillDemo(d.email)}
              className="rounded-md border border-hairline px-3 py-2.5 text-left transition-colors hover:border-gold/50"
            >
              <span className="block font-mono text-xs text-porcelain">{d.email}</span>
              <span className="mt-0.5 block text-xs text-mist">{zh ? d.zh : d.en}</span>
            </button>
          ))}
        </div>
        <p className="mt-3 text-xs text-mist/80">
          {zh
            ? "演示数据仅用于体验，会随环境重置。"
            : "Demo data exists for exploration and resets with the environment."}
        </p>
      </div>
    </div>
  );
}
