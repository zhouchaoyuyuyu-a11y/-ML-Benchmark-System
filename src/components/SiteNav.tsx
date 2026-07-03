"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { navigation } from "@/lib/copy";
import type { Locale } from "@/lib/types";

export default function SiteNav({
  locale,
  userNickname,
  userType,
}: {
  locale: Locale;
  userNickname: string | null;
  userType: string | null;
}) {
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(null);
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    setOpen(false);
    setExpanded(null);
  }, [pathname]);

  async function toggleLocale() {
    await fetch("/api/locale", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ locale: locale === "zh" ? "en" : "zh" }),
    });
    router.refresh();
  }

  const label = (item: { en: string; zh: string }) => (locale === "zh" ? item.zh : item.en);

  return (
    <header className="sticky top-0 z-40 border-b border-hairline bg-ink/90 backdrop-blur">
      <div className="mx-auto flex h-16 w-full max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        <Link href="/" className="flex items-baseline gap-2">
          <span className="font-display text-xl tracking-[0.18em] text-porcelain">ZOTAIX</span>
          <span className="hidden text-xs tracking-[0.3em] text-gold sm:inline">卓序</span>
        </Link>

        <nav className="hidden items-center gap-1 lg:flex">
          {navigation.map((item) => (
            <div key={item.href} className="group relative">
              <Link
                href={item.href}
                className={`rounded-md px-3 py-2 text-sm transition-colors ${
                  pathname.startsWith(item.href) ? "text-gold" : "text-mist hover:text-porcelain"
                }`}
              >
                {label(item)}
                {item.children && <span className="ml-1 text-[10px] text-mist">▾</span>}
              </Link>
              {item.children && (
                <div className="invisible absolute left-0 top-full pt-2 opacity-0 transition-all group-hover:visible group-hover:opacity-100">
                  <div className="zx-card min-w-56 p-2">
                    {item.children.map((child) => (
                      <Link
                        key={child.href}
                        href={child.href}
                        className="block rounded-md px-3 py-2 text-sm text-mist transition-colors hover:bg-veil hover:text-porcelain"
                      >
                        {label(child)}
                      </Link>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <button
            onClick={toggleLocale}
            className="rounded-md border border-hairline px-2.5 py-1.5 text-xs text-mist transition-colors hover:border-gold hover:text-gold"
            aria-label="Switch language"
          >
            {locale === "zh" ? "EN" : "中文"}
          </button>
          {userNickname ? (
            <Link
              href={userType === "admin" ? "/admin" : "/profile"}
              className="hidden items-center gap-2 rounded-md border border-gold/40 px-3 py-1.5 text-xs text-gold sm:inline-flex"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-gold" />
              {userNickname}
            </Link>
          ) : (
            <Link
              href="/login"
              className="hidden rounded-md bg-gold px-4 py-1.5 text-xs font-medium text-ink transition-colors hover:bg-gold-deep hover:text-porcelain sm:inline-flex"
            >
              {locale === "zh" ? "登录 / 注册" : "Sign in"}
            </Link>
          )}
          <button
            onClick={() => setOpen(!open)}
            className="flex h-9 w-9 items-center justify-center rounded-md border border-hairline text-porcelain lg:hidden"
            aria-label="Menu"
          >
            {open ? "✕" : "☰"}
          </button>
        </div>
      </div>

      {open && (
        <div className="border-t border-hairline bg-ink lg:hidden">
          <div className="mx-auto max-w-6xl space-y-1 px-4 py-4">
            {navigation.map((item) => (
              <div key={item.href}>
                <div className="flex items-center justify-between">
                  <Link href={item.href} className="flex-1 rounded-md px-3 py-2.5 text-sm text-porcelain">
                    {label(item)}
                  </Link>
                  {item.children && (
                    <button
                      onClick={() => setExpanded(expanded === item.href ? null : item.href)}
                      className="px-3 py-2 text-mist"
                      aria-label={`Expand ${item.en}`}
                    >
                      {expanded === item.href ? "−" : "+"}
                    </button>
                  )}
                </div>
                {item.children && expanded === item.href && (
                  <div className="ml-3 border-l border-hairline pl-3">
                    {item.children.map((child) => (
                      <Link key={child.href} href={child.href} className="block rounded-md px-3 py-2 text-sm text-mist">
                        {label(child)}
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            ))}
            <div className="pt-2">
              {userNickname ? (
                <Link href={userType === "admin" ? "/admin" : "/profile"} className="block rounded-md border border-gold/40 px-3 py-2.5 text-center text-sm text-gold">
                  {userNickname}
                </Link>
              ) : (
                <Link href="/login" className="block rounded-md bg-gold px-3 py-2.5 text-center text-sm font-medium text-ink">
                  {locale === "zh" ? "登录 / 注册" : "Sign in"}
                </Link>
              )}
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
