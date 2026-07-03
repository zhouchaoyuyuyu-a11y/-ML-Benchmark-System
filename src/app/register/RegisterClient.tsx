"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState, type FormEvent } from "react";
import { Button, Notice } from "@/components/ui";

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

export default function RegisterClient({ zh = false }: { zh?: boolean }) {
  const router = useRouter();
  const params = useSearchParams();
  const next = params.get("next");

  const [nickname, setNickname] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (password.length < 6) {
      setError(zh ? "密码至少需要 6 个字符。" : "Password needs at least 6 characters.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ nickname, email, password }),
      });
      const data = (await res.json().catch(() => ({}))) as { ok?: boolean; error?: string };
      if (!res.ok || !data.ok) {
        setError(data.error ?? (zh ? "注册没有成功，请再试一次。" : "Registration did not go through — please try again."));
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

  return (
    <div className="space-y-5">
      <form onSubmit={submit} className="zx-card p-6 sm:p-8">
        <h2 className="font-display text-xl text-porcelain">{zh ? "创建你的账号" : "Create your account"}</h2>
        <p className="mt-1.5 text-sm text-mist">
          {zh
            ? "一分钟内开启你的秩序中枢——之后填多少、留多久，都由你决定。"
            : "Open your Order Hub in under a minute — what you fill in, and for how long, stays your call."}
        </p>

        <div className="mt-6 space-y-4">
          <label className="block">
            <span className="text-xs uppercase tracking-wider text-mist">{zh ? "昵称" : "Nickname"}</span>
            <input
              type="text"
              autoComplete="nickname"
              value={nickname}
              onChange={(e) => setNickname(e.target.value)}
              placeholder={zh ? "礼宾如何称呼你" : "What the concierge should call you"}
              className={`mt-1.5 ${inputCls}`}
              maxLength={60}
            />
          </label>
          <label className="block">
            <span className="text-xs uppercase tracking-wider text-mist">{zh ? "邮箱" : "Email"}</span>
            <input
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className={`mt-1.5 ${inputCls}`}
            />
          </label>
          <label className="block">
            <span className="text-xs uppercase tracking-wider text-mist">{zh ? "密码" : "Password"}</span>
            <input
              type="password"
              required
              minLength={6}
              autoComplete="new-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={zh ? "至少 6 个字符" : "At least 6 characters"}
              className={`mt-1.5 ${inputCls}`}
            />
          </label>
        </div>

        {error && (
          <div className="mt-4">
            <Notice tone="ember">
              {error}{" "}
              {error.toLowerCase().includes("exists") && (
                <Link href="/login" className="text-gold hover:underline">
                  {zh ? "去登录 →" : "Go to sign in →"}
                </Link>
              )}
            </Notice>
          </div>
        )}

        <Button type="submit" variant="gold" className="mt-6 w-full" disabled={loading}>
          {loading ? (zh ? "创建中…" : "Creating…") : zh ? "免费注册" : "Register free"}
        </Button>

        <p className="mt-4 text-xs leading-relaxed text-mist">
          {zh ? (
            <>
              注册即表示同意
              <Link href="/legal/terms" className="text-gold hover:underline">
                《用户协议》
              </Link>
              与
              <Link href="/legal/privacy" className="text-gold hover:underline">
                《隐私政策》
              </Link>
              。你的档案默认私密，可随时导出或删除。
            </>
          ) : (
            <>
              Registering accepts the{" "}
              <Link href="/legal/terms" className="text-gold hover:underline">
                User Terms
              </Link>{" "}
              and{" "}
              <Link href="/legal/privacy" className="text-gold hover:underline">
                Privacy Policy
              </Link>
              . Your archive is private by default and can be exported or deleted at any time.
            </>
          )}
        </p>
      </form>

      <div className="rounded-lg border border-hairline bg-obsidian/60 p-5">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
          {zh ? "只想先看看？" : "Just looking first?"}
        </p>
        <p className="mt-2 text-xs leading-relaxed text-mist">
          {zh
            ? "你可以先用演示账号体验完整的会员档案、草稿与共创数据，再决定是否注册。"
            : "Try a pre-seeded demo account with full member archives, drafts, and co-creation data before you commit."}
        </p>
        <Link href="/login" className="mt-2 inline-block text-sm text-gold hover:underline">
          {zh ? "查看演示账号 →" : "See the demo accounts →"}
        </Link>
      </div>
    </div>
  );
}
