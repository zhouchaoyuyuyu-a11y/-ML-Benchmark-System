# ZOTAIX — AI Concierge Customization Platform

> **EN** — ZOTAIX is an AI concierge platform that turns emotions, relationships, scenarios, and budgets into bespoke spirits, fragrance directions, bottle design, gifting systems, and digital identity records.
>
> **ZH** — ZOTAIX 卓序是一个 AI 礼宾式定制平台，用 AI 将人的情绪、关系、场景和预算，转化为可确认、可报价、可交付的酒饮、香氛、瓶身、包装与礼赠方案。

This repository contains the full ZOTAIX web platform (Next.js) plus a small legacy ML benchmark course project preserved under `backend/` and `frontend/` (Python/Flask — unrelated to the platform, kept intact: 本仓库同时保留了机器学习实践课程的 Benchmark 系统代码).

## Platform overview

The product is built on a five-module object chain:

| Module | Route | Role |
| --- | --- | --- |
| **Forge** | `/forge`, `/concierge` | AI concierge & task orchestration: collects structured intent (emotion, recipient, scenario, budget), produces structured proposals, consumes quota, writes results into business objects |
| **Studio** | `/studio` | Visual & 3D preview: bottle configurator, labels, packaging, fragrance mood boards, emotional cards, digital badges |
| **Design** | `/design` | Structured proposal & version system: names, copy, liquid/fragrance directions, visual styles, versioned with hashes |
| **Trade** | `/trade`, `/market` | Quotes, creative-rights authorization, co-creation rights, enterprise inquiry, human concierge workflow (explicitly **not** an alcohol resale marketplace) |
| **Reserve** | `/reserve` | Long-term archive: ZOTAIX IDs, QR/NFC binding, certificates, aftercare, replenishment, membership |

Core principle: **users create and save a personalized object first**, then decide whether it becomes a real spirit, fragrance, bottle, gift box, enterprise gift, co-creation project, or concierge request. There is no cart on the front door.

Two delivery lines share the chain:

- **Maison ZOTAIX** (`/maison`) — premium/enterprise gifting: AI concierge + human confirmation + quotation + delivery + aftercare + Reserve identity.
- **ZOTAIX Supply** (`/supply`) — mass emotional line: emotional spirits, fragrance supplies, zero-proof rations, labels, emotional cards, low-barrier co-creation.

## Tech stack

- **Next.js 15** (App Router, TypeScript strict) — deploys to **Vercel** with zero special configuration
- **Tailwind CSS v4** design tokens (`src/app/globals.css`)
- **Data layer**: complete Postgres/Supabase schema in `db/schema.sql` + `db/seed.sql`; at runtime a **seeded in-memory store** (`src/lib/store.ts`) keeps every feature functional when `DATABASE_URL` is not configured (graceful-fallback design; state is per server instance)
- **AI layer**: Anthropic or OpenAI via environment variables (`src/lib/ai/provider.ts`); a deterministic **Atelier engine** (`src/lib/ai/engine.ts`) guarantees structured output when no key is configured or a provider call fails
- **PWA**: `public/manifest.webmanifest`, `public/sw.js` (offline fallback at `/offline`), install prompt component
- **App packaging**: Capacitor wrapper config in `app-wrapper/`, store templates in `app-store-assets/`, guide in `docs/APP_PACKAGING.md`

## Local development

```bash
npm install
npm run dev        # http://localhost:3000
npm run build      # production build
npm run typecheck  # tsc --noEmit
```

No environment variables are required to run locally — every integration degrades gracefully (see below). Copy `.env.example` to `.env.local` to configure real services.

### Demo accounts (seeded)

| Account | Password | Role |
| --- | --- | --- |
| `admin@zotaix.demo` | `zotaix-demo` | Admin console (`/admin`) |
| `member@zotaix.demo` | `zotaix-demo` | Core Sequence Pro member |
| `lite@zotaix.demo` | `zotaix-demo` | Core Sequence Lite member |
| `user@zotaix.demo` | `zotaix-demo` | Free registered user |

Guests get a small daily AI trial without an account; registering unlocks profile, drafts, and Reserve.

## Environment variables

All secrets come from environment variables — **nothing is hardcoded**. See `.env.example` for the full annotated list. Summary:

| Group | Variables | Behavior when unset |
| --- | --- | --- |
| Site | `NEXT_PUBLIC_SITE_URL`, `SESSION_SECRET`, `ADMIN_EMAIL` | Sensible defaults; set both in production |
| Database | `DATABASE_URL` | Seeded in-memory store (demo mode; resets per instance) |
| AI | `AI_PROVIDER`, `ANTHROPIC_API_KEY`/`ANTHROPIC_MODEL`, `OPENAI_API_KEY`/`OPENAI_MODEL` | Deterministic Atelier engine generates all structured proposals |
| Payments | `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `WECHAT_PAY_MCH_ID`, `WECHAT_PAY_KEY`, `ALIPAY_APP_ID`, `ALIPAY_PRIVATE_KEY`, `PAYPAL_CLIENT_ID`, `PAYPAL_SECRET` | Orders recorded in **test mode** and routed to human-concierge confirmation; nothing crashes, nothing pretends to be live |
| WeChat | `WECHAT_APP_ID`, `WECHAT_APP_SECRET` | `/wechat` page + admin config still work; menu/auto-reply stored for publishing once credentials exist |
| Storage | `STORAGE_BUCKET`, `CDN_URL` | Local `/public` assets + generated SVG marks |
| App | `APP_IOS_URL`, `APP_ANDROID_URL`, `APP_APK_URL` | Download page offers PWA install and explains store distribution is configured in admin |

## Deployment (Vercel)

1. Import the repository into Vercel (root directory = repo root; framework auto-detected as Next.js).
2. Set environment variables in **Vercel → Project → Settings → Environment Variables** (at minimum `NEXT_PUBLIC_SITE_URL` and `SESSION_SECRET`; add AI/payment/WeChat keys as they become available).
3. Deploy. `next build` is the only build step; no native dependencies.

To use a real database: create a Postgres instance (e.g. Supabase), run `db/schema.sql` then `db/seed.sql`, and set `DATABASE_URL`. The schema includes commented Supabase RLS statements.

## PWA & app wrapping

- PWA works out of the box: manifest + service worker + offline page + install prompt. Verify with Lighthouse after deployment.
- Native wrappers: see `docs/APP_PACKAGING.md` for the Capacitor remote-URL wrapper flow (iOS + Android), deep links / Universal Links / App Links samples, icons/splash, and store submission checklist. Store listing copy templates live in `app-store-assets/`.
- App distribution URLs (App Store / Google Play / APK) are configured at `/admin/app` and render on `/download`. The page never claims a store listing exists until a URL is configured.

## WeChat Official Account

`/wechat` is the public page; `/admin/wechat` is the console. Configuration stored per environment:

- Account name, QR code image URL, customer-service URL, enabled state
- Menu structure (3 groups: AI Concierge / Co-Creation & Membership / Premium Customization)
- Auto-replies (follow reply + keyword replies: 情绪 / 礼物 / 共创 / 高定 / APP / 客服)
- Real credentials (`WECHAT_APP_ID`, `WECHAT_APP_SECRET`) come only from environment variables; the admin page shows configured/not-configured state and never displays secrets. Menu and replies are stored so they can be published to the WeChat platform once a certified account is connected.

## Global social media

`/social` renders the account matrix (Instagram, TikTok, X, YouTube, LinkedIn, Facebook, Pinterest, Threads) from the store; `/admin/social` edits URL, tracking parameters, order, and enabled state per platform. `/admin/content-calendar` manages the cross-platform content calendar (draft → scheduled → published).

## Admin console

`/admin` (admin account required) includes: dashboard, users, user profiles, memberships, conversations, AI usage & cost logs, object drafts, design versions, Reserve records, trade & inquiries (with human-concierge leads), co-creation review, orders & payments, moderation, WeChat config, social config, app & download config, content calendar, CMS blocks, legal page registry, and global settings (quotas, prices, co-creation thresholds, age gate).

Review actions available on reviewable objects: approve / reject / request revision / escalate to concierge / mark compliance risk / mark supply-chain infeasible / feature / hide / unpublish.

## Compliance model

- AI outputs are **creative proposals**; physical delivery requires human confirmation, supply-chain confirmation, age & region compliance checks, and final quotation. This notice renders on every relevant page.
- Small-order customization = standard base liquid + personalized expression (label copy, visual style, story card, serial, QR/NFC, badge, Reserve record) — never per-bottle formulas or molds.
- Co-creation thresholds: 10 people → public page; 30 → platform review; 50 bottles → label/gift-box theming; 100 → flavor review; 300 → enterprise review; 500+ → packaging & supply chain; 1000+ → partnership review.
- Trade prohibits user-to-user alcohol resale, unreviewed vouchers, unauthorized sellers, and off-platform transaction guidance.
- Full legal set under `/legal/*`: terms, privacy, cookies, AI notice, alcohol compliance, minor protection, membership agreement, co-creation rules, trade rules, reserve rules, app privacy, contact.

## Repository layout

```
src/app/            App Router pages (public + /admin + /api)
src/components/     Shared UI (design system, nav, footer, proposal cards, admin shell)
src/lib/            Types, seeded store, auth, quota, AI provider + Atelier engine, SEO, i18n
db/                 schema.sql + seed.sql (Postgres / Supabase)
public/             PWA manifest, service worker, icons
app-store-assets/   Store listing templates (names, descriptions, keywords, screenshots, privacy)
app-wrapper/        Capacitor configuration for iOS/Android wrapping
docs/               APP_PACKAGING.md, AGENT_CONTRACT.md (internal implementation contract)
backend/, frontend/ Legacy ML benchmark course project (unrelated; preserved)
```
