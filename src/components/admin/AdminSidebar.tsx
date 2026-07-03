"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const GROUPS: { title: string; links: { href: string; label: string }[] }[] = [
  {
    title: "Overview",
    links: [{ href: "/admin/dashboard", label: "Dashboard" }],
  },
  {
    title: "People",
    links: [
      { href: "/admin/users", label: "Users" },
      { href: "/admin/profiles", label: "User Profiles" },
      { href: "/admin/memberships", label: "Memberships" },
    ],
  },
  {
    title: "AI",
    links: [
      { href: "/admin/conversations", label: "Conversations" },
      { href: "/admin/ai-usage", label: "AI Usage & Costs" },
    ],
  },
  {
    title: "Objects",
    links: [
      { href: "/admin/drafts", label: "Object Drafts" },
      { href: "/admin/designs", label: "Design Versions" },
      { href: "/admin/reserve", label: "Reserve Records" },
    ],
  },
  {
    title: "Commerce",
    links: [
      { href: "/admin/trade", label: "Trade & Inquiries" },
      { href: "/admin/co-create", label: "Co-Creation Review" },
      { href: "/admin/orders", label: "Orders & Payments" },
      { href: "/admin/moderation", label: "Moderation" },
    ],
  },
  {
    title: "Channels",
    links: [
      { href: "/admin/wechat", label: "WeChat" },
      { href: "/admin/social", label: "Social Media" },
      { href: "/admin/app", label: "App & Downloads" },
      { href: "/admin/content-calendar", label: "Content Calendar" },
    ],
  },
  {
    title: "Content",
    links: [
      { href: "/admin/cms", label: "CMS & Homepage" },
      { href: "/admin/legal", label: "Legal Pages" },
      { href: "/admin/settings", label: "Global Settings" },
    ],
  },
];

export default function AdminSidebar({ nickname }: { nickname: string }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <>
      <div className="flex items-center justify-between border-b border-hairline px-4 py-3 lg:hidden">
        <p className="font-display text-sm tracking-[0.2em] text-porcelain">ZOTAIX ADMIN</p>
        <button onClick={() => setOpen(!open)} className="rounded-md border border-hairline px-3 py-1.5 text-xs text-mist">
          {open ? "Close" : "Menu"}
        </button>
      </div>
      <aside className={`${open ? "block" : "hidden"} w-full shrink-0 border-r border-hairline bg-obsidian/60 lg:block lg:w-60`}>
        <div className="hidden border-b border-hairline px-5 py-4 lg:block">
          <p className="font-display text-sm tracking-[0.2em] text-porcelain">ZOTAIX ADMIN</p>
          <p className="mt-1 text-xs text-mist">{nickname}</p>
        </div>
        <nav className="space-y-4 px-3 py-4">
          {GROUPS.map((g) => (
            <div key={g.title}>
              <p className="px-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-mist">{g.title}</p>
              <div className="mt-1 space-y-0.5">
                {g.links.map((l) => (
                  <Link
                    key={l.href}
                    href={l.href}
                    className={`block rounded-md px-2 py-1.5 text-sm transition-colors ${
                      pathname === l.href ? "bg-veil text-gold" : "text-mist hover:bg-veil/60 hover:text-porcelain"
                    }`}
                  >
                    {l.label}
                  </Link>
                ))}
              </div>
            </div>
          ))}
          <div className="px-2 pt-2">
            <button
              type="button"
              className="text-xs text-mist hover:text-ember"
              onClick={async () => {
                await fetch("/api/auth/logout", { method: "POST" });
                window.location.href = "/";
              }}
            >
              Sign out
            </button>
          </div>
        </nav>
      </aside>
    </>
  );
}
