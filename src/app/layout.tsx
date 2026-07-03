import type { Metadata, Viewport } from "next";
import "./globals.css";
import SiteNav from "@/components/SiteNav";
import SiteFooter from "@/components/SiteFooter";
import PWARegister from "@/components/PWARegister";
import JsonLd from "@/components/JsonLd";
import { getLocale } from "@/lib/locale";
import { getSessionUser } from "@/lib/auth";
import { organizationJsonLd, pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export const metadata: Metadata = {
  ...pageMetadata({
    title: "AI Concierge Customization Platform",
    description:
      "ZOTAIX turns emotions, relationships, scenarios, and budgets into bespoke spirits, fragrance directions, bottle design, gifting systems, and digital identity records.",
    path: "",
  }),
  applicationName: "ZOTAIX",
  manifest: "/manifest.webmanifest",
  appleWebApp: { capable: true, title: "ZOTAIX", statusBarStyle: "black-translucent" },
  icons: {
    icon: [{ url: "/icons/icon-192.png", sizes: "192x192" }, { url: "/icons/icon-512.png", sizes: "512x512" }],
    apple: "/icons/apple-touch-icon.png",
  },
};

export const viewport: Viewport = {
  themeColor: "#0a0c11",
  width: "device-width",
  initialScale: 1,
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const locale = await getLocale();
  const user = await getSessionUser();
  const appConfig = db().app_config;

  return (
    <html lang={locale === "zh" ? "zh-CN" : "en"}>
      <body className="flex min-h-screen flex-col">
        <JsonLd data={organizationJsonLd()} />
        <SiteNav locale={locale} userNickname={user?.nickname ?? null} userType={user?.user_type ?? null} />
        <main className="flex-1">{children}</main>
        <SiteFooter locale={locale} />
        <PWARegister enabled={appConfig.pwa_enabled && appConfig.install_prompt_enabled} />
      </body>
    </html>
  );
}
