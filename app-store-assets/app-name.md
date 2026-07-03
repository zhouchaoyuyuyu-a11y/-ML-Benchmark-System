# App name, subtitle, category, age rating

## Name

| Field | Value | Limit |
| --- | --- | --- |
| App Store name | `ZOTAIX` | 30 chars — 6 used |
| Google Play title | `ZOTAIX` | 30 chars — 6 used |
| Bundle display name (under icon) | `ZOTAIX` | keep ≤ 12 chars so it never truncates |

The Chinese brand form `ZOTAIX 卓序` is used inside descriptions and keywords,
not in the store title, so the icon label stays identical across locales.

## Subtitle (App Store) / Short tagline

All variants are ≤ 30 characters (App Store subtitle hard limit).

**English (pick one):**

| Variant | Chars |
| --- | --- |
| `Emotions, made into objects` | 27 |
| `AI concierge for bespoke gifts` | 30 |
| `Bespoke spirits & fragrance` | 27 |

Recommended: `Emotions, made into objects` — it states the platform's core
principle (create the object first) without leaning only on gifting.

**Chinese (pick one):**

| Variant | Chars |
| --- | --- |
| `把情绪做成可收藏的对象` | 11 |
| `AI 礼宾 · 定制酒香与礼赠` | 14 |
| `先创造对象，再决定成真` | 11 |

Recommended: `把情绪做成可收藏的对象`.

## Category

| Store | Primary | Secondary |
| --- | --- | --- |
| App Store | Lifestyle | Shopping |
| Google Play | Lifestyle | Shopping (tag) |

Rationale: the core loop (emotional check-in → personalized object → archive)
is a lifestyle experience; commerce is the optional second step, so Shopping
stays secondary.

## Age rating

The platform references and customizes alcoholic beverages (spirits, wine),
alongside a zero-proof Supply line. Rate for the alcohol references, not the
zero-proof subset.

### App Store (Apple age rating questionnaire)

- **Alcohol, Tobacco, or Drug Use or References: Frequent/Intense** → app
  rating **17+**.
- All other categories: None.
- Do not answer "Infrequent/Mild": bottle design, liquid directions, and
  spirit customization are core features, and an under-declared rating is a
  common rejection reason.

### Google Play (IARC questionnaire)

- Declare **references to alcohol** → typical outcomes: PEGI 16 (Europe),
  USK 16 (Germany), ESRB Teen/Mature 17+ (Americas), ACB M (Australia).
- Declare **user-generated content with moderation** (public co-creation
  projects and public Reserve pages pass human review — see
  `moderation_logs` in the platform).
- Declare that the app does **not** sell alcohol directly in-app: physical
  delivery is confirmed by a human concierge with age and region checks.

### In-app gate

Independent of store rating, the platform enforces an **18+ age gate** on
alcohol-related surfaces (`age_gate_enabled` platform setting; AgeGate
component on maison/forge/trade pages). State this in review notes — it
speeds up approval.

### Regional notes

- **Mainland China:** distribute via APK / regional stores with the same 18+
  gate; alcohol advertising rules prohibit health claims and depictions of
  drinking acts — ZOTAIX copy already avoids both (directions and design
  only, never consumption imagery).
- **Middle East (KSA, UAE, Kuwait, etc.):** consider excluding storefronts
  where alcohol-related apps are restricted, or ship the Supply
  (zero-proof) positioning for those storefronts after legal review.
- **South Korea / Japan:** 17+/18+ declaration is sufficient; no separate
  liquor-license requirement because the app itself does not transact
  alcohol — orders route to human-confirmed regional fulfillment.
- **United States:** 17+ with in-app 21+ verification handled at the
  fulfillment step by the human concierge (stated in review notes).

### Review notes (paste into both consoles)

> ZOTAIX generates AI creative proposals (liquid direction, fragrance
> direction, bottle design, label copy). It does not sell alcohol in-app.
> Physical production and delivery require human concierge confirmation with
> age verification and regional compliance checks. An 18+ age gate covers all
> alcohol-related screens. A zero-proof product line exists but the rating is
> declared on the alcohol references.
