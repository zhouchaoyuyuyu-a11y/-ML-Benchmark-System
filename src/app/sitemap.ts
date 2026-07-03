import type { MetadataRoute } from "next";
import { siteUrl } from "@/lib/config";
import { db } from "@/lib/store";

export default function sitemap(): MetadataRoute.Sitemap {
  const staticPaths = [
    "",
    "/concierge",
    "/supply",
    "/maison",
    "/forge",
    "/studio",
    "/design",
    "/trade",
    "/reserve",
    "/co-create",
    "/market",
    "/membership",
    "/download",
    "/wechat",
    "/social",
    "/about",
    "/cases",
    "/blog",
    "/legal/privacy",
    "/legal/terms",
    "/legal/cookies",
    "/legal/ai",
    "/legal/alcohol",
    "/legal/minors",
    "/legal/membership",
    "/legal/co-create",
    "/legal/trade",
    "/legal/reserve",
    "/legal/app",
    "/legal/contact",
  ];

  const data = db();
  const dynamic = [
    ...data.case_studies.map((c) => `/cases/${c.slug}`),
    ...data.blog_posts.map((p) => `/blog/${p.slug}`),
    ...data.co_creation_projects.filter((p) => p.public_visible).map((p) => `/co-create/${p.id}`),
    ...data.reserve_records.filter((r) => r.privacy_level === "public").map((r) => `/reserve/${r.id}`),
  ];

  return [...staticPaths, ...dynamic].map((path) => ({
    url: `${siteUrl}${path}`,
    changeFrequency: path === "" ? "daily" : "weekly",
    priority: path === "" ? 1 : 0.7,
  }));
}
