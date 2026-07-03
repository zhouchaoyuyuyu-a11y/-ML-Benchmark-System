// Central environment access. Never hardcode secrets — every integration is
// driven by environment variables and degrades gracefully when missing.

export const siteUrl =
  process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/$/, "") || "https://zotaix-web.vercel.app";

export const siteName = "ZOTAIX";

function has(v?: string): boolean {
  return typeof v === "string" && v.trim().length > 0;
}

export interface IntegrationStatus {
  key: string;
  label: string;
  configured: boolean;
  envVars: string[];
  fallback: string;
}

export function integrationStatus(): IntegrationStatus[] {
  return [
    {
      key: "database",
      label: "Database (Postgres / Supabase)",
      configured: has(process.env.DATABASE_URL),
      envVars: ["DATABASE_URL"],
      fallback: "Seeded in-memory store (resets per deployment instance).",
    },
    {
      key: "ai",
      label: "AI provider",
      configured: has(process.env.ANTHROPIC_API_KEY) || has(process.env.OPENAI_API_KEY),
      envVars: ["AI_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"],
      fallback: "Deterministic ZOTAIX Atelier engine (structured local generation).",
    },
    {
      key: "stripe",
      label: "Stripe",
      configured: has(process.env.STRIPE_SECRET_KEY),
      envVars: ["STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET"],
      fallback: "Orders recorded in test mode; human concierge confirmation.",
    },
    {
      key: "wechat",
      label: "WeChat Official Account",
      configured: has(process.env.WECHAT_APP_ID) && has(process.env.WECHAT_APP_SECRET),
      envVars: ["WECHAT_APP_ID", "WECHAT_APP_SECRET"],
      fallback: "QR display + configuration preview; menu/auto-reply stored for publishing.",
    },
    {
      key: "wechat_pay",
      label: "WeChat Pay",
      configured: has(process.env.WECHAT_PAY_MCH_ID) && has(process.env.WECHAT_PAY_KEY),
      envVars: ["WECHAT_PAY_MCH_ID", "WECHAT_PAY_KEY"],
      fallback: "Orders routed to human concierge confirmation.",
    },
    {
      key: "alipay",
      label: "Alipay",
      configured: has(process.env.ALIPAY_APP_ID) && has(process.env.ALIPAY_PRIVATE_KEY),
      envVars: ["ALIPAY_APP_ID", "ALIPAY_PRIVATE_KEY"],
      fallback: "Orders routed to human concierge confirmation.",
    },
    {
      key: "paypal",
      label: "PayPal",
      configured: has(process.env.PAYPAL_CLIENT_ID) && has(process.env.PAYPAL_SECRET),
      envVars: ["PAYPAL_CLIENT_ID", "PAYPAL_SECRET"],
      fallback: "Orders routed to human concierge confirmation.",
    },
    {
      key: "storage",
      label: "Object storage / CDN",
      configured: has(process.env.STORAGE_BUCKET),
      envVars: ["STORAGE_BUCKET", "CDN_URL"],
      fallback: "Local /public assets and generated SVG placeholders.",
    },
  ];
}

export function aiProvider(): "anthropic" | "openai" | "atelier" {
  const pref = (process.env.AI_PROVIDER || "").toLowerCase();
  if (pref === "anthropic" && has(process.env.ANTHROPIC_API_KEY)) return "anthropic";
  if (pref === "openai" && has(process.env.OPENAI_API_KEY)) return "openai";
  if (has(process.env.ANTHROPIC_API_KEY)) return "anthropic";
  if (has(process.env.OPENAI_API_KEY)) return "openai";
  return "atelier";
}

export const adminEmail = process.env.ADMIN_EMAIL || "admin@zotaix.demo";

export const appLinks = {
  ios: process.env.APP_IOS_URL || "",
  android: process.env.APP_ANDROID_URL || "",
  apk: process.env.APP_APK_URL || "",
};

export const sessionSecret =
  process.env.SESSION_SECRET || "zotaix-dev-secret-change-in-production";
