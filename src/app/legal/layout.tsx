import Link from "next/link";
import type { ReactNode } from "react";
import { Section } from "@/components/ui";
import { footerLegal } from "@/lib/copy";
import { getLocale } from "@/lib/locale";

export default async function LegalLayout({ children }: { children: ReactNode }) {
  const locale = await getLocale();
  const zh = locale === "zh";

  return (
    <div className="border-b border-hairline">
      <Section className="py-10 sm:py-16">
        <div className="grid gap-10 lg:grid-cols-[240px_minmax(0,1fr)]">
          {/* Side navigation: all legal documents */}
          <aside className="lg:sticky lg:top-24 lg:self-start">
            <p className="text-xs font-semibold uppercase tracking-[0.25em] text-gold">
              {zh ? "法律与政策" : "Legal & Policies"}
            </p>
            <p className="mt-2 text-xs leading-relaxed text-mist">
              {zh
                ? "ZOTAIX 卓序平台的完整规则文档。所有版本与生效日期均在页首标明。"
                : "The complete rulebook of the ZOTAIX platform. Version and effective date are stated at the top of each document."}
            </p>
            <nav
              aria-label={zh ? "法律文档导航" : "Legal documents"}
              className="mt-5 flex flex-wrap gap-2 lg:flex-col lg:gap-0"
            >
              {footerLegal.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="rounded-md border border-hairline px-3 py-1.5 text-xs text-mist transition-colors hover:border-gold/50 hover:text-gold lg:border-0 lg:border-l lg:border-hairline lg:rounded-none lg:px-3 lg:py-2 lg:text-sm lg:hover:border-gold"
                >
                  {zh ? item.zh : item.en}
                </Link>
              ))}
            </nav>
          </aside>

          {/* Document column */}
          <div className="min-w-0 max-w-3xl">{children}</div>
        </div>
      </Section>
    </div>
  );
}
