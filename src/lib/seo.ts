import type { Metadata } from "next";
import { siteName, siteUrl } from "./config";

export interface PageSeo {
  title: string;
  description: string;
  path: string;
  image?: string;
  keywords?: string[];
}

/** Standard metadata for a public page: title, description, canonical, OG, Twitter. */
export function pageMetadata({ title, description, path, image, keywords }: PageSeo): Metadata {
  const url = `${siteUrl}${path}`;
  const ogImage = image ?? `${siteUrl}/api/og?title=${encodeURIComponent(title)}`;
  return {
    title: `${title} · ${siteName}`,
    description,
    keywords,
    alternates: { canonical: url, languages: { en: url, "zh-CN": url } },
    openGraph: {
      title: `${title} · ${siteName}`,
      description,
      url,
      siteName,
      type: "website",
      images: [{ url: ogImage, width: 1200, height: 630 }],
      locale: "en_US",
      alternateLocale: ["zh_CN"],
    },
    twitter: {
      card: "summary_large_image",
      title: `${title} · ${siteName}`,
      description,
      images: [ogImage],
    },
  };
}

export function organizationJsonLd() {
  return {
    "@context": "https://schema.org",
    "@type": "Organization",
    name: siteName,
    url: siteUrl,
    description:
      "AI concierge platform that turns emotions, relationships, scenarios, and budgets into bespoke spirits, fragrance directions, bottle design, gifting systems, and digital identity records.",
    logo: `${siteUrl}/icons/icon-512.png`,
  };
}
