# ZOTAIX implementation contract (for page implementers)

Repo root: `/home/user/-ML-Benchmark-System`. Next.js 15 App Router + TypeScript strict + Tailwind CSS v4. Path alias `@/*` → `src/*`. Do NOT touch files outside your assigned list. Do NOT edit shared foundation files.

## Design tokens (Tailwind utilities available)
Colors: `bg-ink` (page bg, #0a0c11), `bg-obsidian`, `bg-veil`, `border-hairline`, `text-porcelain` (primary), `text-mist` (muted), `text-gold`/`bg-gold`/`border-gold` (premium accent), `text-supply` (playful accent, Supply line), `text-jade` (success), `text-ember` (error/risk). Serif headings: className `font-display`. Helper CSS classes: `zx-card`, `zx-card-hover`, `zx-meridian`, `zx-grid-bg`, `zx-skeleton`, `zx-rotate-slow`, `zx-fade-up`.
Tone: premium, restrained, technological, orderly. Supply/membership areas may be slightly playful (use `supply` accent). Maison/enterprise areas stay restrained (gold accent).

## Shared components — import and use EXACTLY these signatures

From `@/components/ui` (server-safe):
- `Section({children, className?, id?})` — max-w-6xl container
- `SectionHeader({eyebrow?, title, description?, className?})`
- `PageHero({eyebrow?, title, description?, children?, tone?: "default"|"supply"|"maison"})` — children = CTA row
- `ButtonLink({href, children, variant?: "gold"|"outline"|"ghost"|"supply"|"danger", className?})`
- `Button({variant?, ...buttonProps})`
- `Card({children, className?, hover?})`
- `Tag({children, tone?: "default"|"gold"|"supply"|"jade"|"ember"})`
- `StatusPill({status: string})` — colors common statuses automatically
- `Notice({children, tone?: "gold"|"ember"|"supply", title?})` — compliance notices
- `EmptyState({title, description, action?})`
- `Stat({label, value, hint?})`
- `Meridian({className?})` — gold divider line
- `ProgressBar({value, max, tone?: "gold"|"supply"})`
- `DefinitionRow({label, children})` — dt/dd row for detail pages

Other shared components:
- `ProposalCard` (default, CLIENT) from `@/components/ProposalCard`: `{proposal: ConceptProposal, zh?: boolean, compact?: boolean}` — renders structured AI result + next-action buttons (save/card/share/co-create/casting/concierge built in).
- `AgeGate` (default, CLIENT) from `@/components/AgeGate`: `{zh?: boolean}` — include on alcohol-related pages (maison, forge, trade, market detail).
- `QRCodeBox` (default, server-safe) from `@/components/QRCodeBox`: `{seed: string, label?: string, size?: number}` — deterministic QR-style SVG identity mark.
- `JsonLd` (default) from `@/components/JsonLd`: `{data: object}`.
- Admin only: `ReviewActions` (default, CLIENT) from `@/components/admin/ReviewActions`: `{targetType: "co_creation_project"|"trade_request"|"moderation_log"|"object_draft"|"content_calendar", targetId: string, actions?: string[]}` — action keys: approve, reject, revision, escalate, feature, hide, unpublish, compliance_risk, infeasible.

## Shared lib

- `@/lib/types` — all entity interfaces (User, ObjectDraft, ReserveRecord, CoCreationProject, TradeRequest, ConceptProposal, AiResult, etc.). Read this file before writing pages.
- `@/lib/store`: `db(): Database` (seeded collections: users, user_profiles, relationship_profiles, conversations, messages, ai_usage_logs, object_drafts, design_versions, memberships, co_creation_projects, co_creation_members, reserve_records, trade_requests, moderation_logs, social_accounts, wechat_config, app_config, content_calendar, orders, concierge_leads, case_studies, blog_posts, cms_blocks, legal_docs, settings). Also `dataMode()`, `newId(prefix)`, `now()`.
- `@/lib/auth`: `getSessionUser(): Promise<User|null>` (server only).
- `@/lib/locale`: `getLocale(): Promise<"en"|"zh">` (server only), `tt(locale, en, zh)`.
- `@/lib/copy`: `brand`, `headline`, `subheadline`, `complianceNotice`, `profileNotice`, `orderWorld` (vocab mapping), `navigation`, `footerLegal`, `pick(locale, {en,zh})`.
- `@/lib/seo`: `pageMetadata({title, description, path, image?, keywords?}): Metadata` — REQUIRED on every public page: `export const metadata = pageMetadata({...})`. Admin pages skip this (layout sets noindex).
- `@/lib/config`: `integrationStatus()`, `aiProvider()`, `appLinks`, `siteUrl`, `adminEmail`.
- `@/lib/quota`: `checkQuota(user, visitorId, tier)`, tiers: "chat" | "proposal" | "creative".

## API endpoints (already implemented — do not recreate)

- `POST /api/ai/generate` body `{mode: "daily"|"gift"|"spirit"|"fragrance"|"copy"|"style"|"recipient"|"co_create"|"enterprise", message, emotion?, recipient?, scenario?, budget?, style?, locale?, conversationId?}` → `{ok, conversationId, result: {reply, proposal, model, tokens_used, quota_remaining, fallback}}`; 429 with `{quota: true, error, upgradeHint}` when quota exhausted; 401 never (guests allowed for mode=daily).
- `POST /api/auth/register` `{email, password, nickname}`; `POST /api/auth/login` `{email, password}` → `{ok, user:{id, nickname, user_type}}`; `POST /api/auth/logout`.
- `POST /api/drafts` `{proposal?, intent?, object_type?, title?, scene?, recipient?, budget?, emotion_tags?, label_copy?, visual_style?, liquid_direction?, scent_direction?, public_visible?}` → `{ok, draft}` (401 for guests). `GET /api/drafts` → `{ok, drafts}`.
- `POST /api/reserve` `{draftId, privacy_level?}` → `{ok, record, note?}`.
- `POST /api/co-create` `{action: "create"|"join"|"vote", projectId?, quantity?, title?, concept?, product_type?, target_quantity?, emotion_tags?}` → create requires membership (403 with upgradeHint), join requires account, vote is open.
- `POST /api/trade` `{request_type: "quote"|"authorization"|"enterprise"|"collaboration"|"replenishment", name?, organization?, contact?, quantity?, budget?, deadline?, delivery_region?, liquid_direction?, scent_direction?, bottle_direction?, packaging_direction?, sample_path?, invoice_required?, logistics_notes?, notes?, draftId?, scenario?}` → `{ok, request, note}` (enterprise/collaboration allowed without login if contact given).
- `POST /api/profile` (see route for fields), `GET /api/profile`; `POST /api/membership/subscribe` `{plan: "lite"|"pro", cycle: "month"|"quarter", method?}`.
- `POST /api/admin/review` `{targetType, targetId, action}` (admin session required).
- `POST /api/admin/config` `{section: "wechat"|"social"|"app"|"settings"|"cms"|"calendar", patch}` (admin session required).
- `GET /api/card?copy=&mark=&keywords=` → PNG emotional card. `GET /api/og?title=&subtitle=` → PNG OG image.

## Page conventions

1. Pages are **async server components**: `const locale = await getLocale(); const zh = locale === "zh";` and read data via `db()`. Interactive parts go in a sibling client component file (e.g. `ConciergeClient.tsx`) marked `"use client"`.
2. Every public page exports `metadata = pageMetadata({title, description, path: "/<route>"})`.
3. Bilingual: page hero title/description and key labels use `zh ? "中文" : "English"`. Body copy may be English-first with Chinese where valuable. Do not build new i18n machinery.
4. Client components must NOT import `@/lib/store`, `@/lib/auth`, `@/lib/locale`, or `@/lib/seo` (server-only). Pass data as props; mutate via fetch to APIs above.
5. Responsive: mobile-first, use `sm:`/`lg:` variants; grids collapse to one column on mobile.
6. FORBIDDEN in any user-facing text: “coming soon”, “future phase”, “later”, “MVP”, “to be implemented”, “placeholder”, “TODO”, “work in progress”, "under construction". Every page must be a complete experience. When an external credential is missing, present the graceful fallback as designed behavior (e.g. “Orders are confirmed by a human concierge” / test mode), never as missing work.
7. Alcohol compliance: any page that promises physical alcohol delivery must include the `complianceNotice` via `<Notice>` and mention age/region checks. Never promise per-bottle new formulas/molds, AI-controlled production, or shipment without human review.
8. Loading/error states: for pages with client-side fetches, include loading (`zx-skeleton` blocks or button spinners), error, empty, and success states.
9. TypeScript strict must pass: no implicit any; unused imports will fail review — keep imports minimal. Apostrophes in JSX text are fine (no ESLint), but keep JSX valid.
10. `next/image` is NOT configured for remote images — use inline SVG, QRCodeBox, or /api/card images with plain `<img>`.
11. Escape `"` inside JSX attribute strings normally; prefer typographic quotes in copy where natural.
12. Forms: controlled inputs styled like: `className="w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none"`.

## Return format
Your final message: list of files created + one line per file describing what it renders. Note anything that deviates from this contract.
