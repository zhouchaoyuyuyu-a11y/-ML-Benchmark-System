# Screenshots — sizes, shot list, capture guide

## Required sizes

### App Store (upload per device class; portrait)

| Device class | Pixels | Notes |
| --- | --- | --- |
| iPhone 6.7" | **1290 × 2796** | iPhone 15 Pro Max class — required |
| iPhone 6.5" | **1284 × 2778** or **1242 × 2688** | iPhone 11 Pro Max / XS Max class |
| iPhone 5.5" | **1242 × 2208** | iPhone 8 Plus class — required if supporting older devices |
| iPad Pro 12.9" | **2048 × 2732** | Required when the iPad device family is enabled |

3–10 screenshots per class; the first two carry almost all conversion weight.

### Google Play

| Asset | Pixels | Notes |
| --- | --- | --- |
| Phone screenshots | **1080 × 1920** minimum (portrait 9:16) | 2–8 images; PNG or JPEG, ≤ 8 MB each |
| 7"/10" tablet | 1200 × 1920 / 1600 × 2560 recommended | Optional but improves tablet placement |
| Feature graphic | **1024 × 500** | Required; shown atop the listing and in promos |
| App icon | 512 × 512 | Exported from `public/icons/icon-512.png` |

## Canonical shot list (same order in every locale)

| # | Screen | Route to capture | Caption EN | Caption ZH |
| --- | --- | --- | --- | --- |
| 1 | **Concierge** | `/concierge` with a completed daily check-in | "Tell it one sentence about today" | 「用一句话，说说今天」 |
| 2 | **Proposal card** | Concierge result showing a structured ConceptProposal (emotional signal, keywords, liquid/scent direction, names, label copy, next actions) | "Get a structured creative proposal" | 「获得结构化的创意提案」 |
| 3 | **Studio bottle** | `/studio` design-version view (e.g. "v2 · Gold Meridian" payload: bottle, label, packaging, liquid) | "Design the bottle, label, and box" | 「设计瓶身、标签与包装」 |
| 4 | **Co-create** | `/co-create` pool showing "Order 03:00" at 64/100 with progress bar | "Co-create — 10 people open a page" | 「共创铸造池 · 十人成页」 |
| 5 | **Reserve certificate** | `/reserve/rsv_001` certificate: ZOTAIX ID `ZX-2026-0611-0001`, QR mark, emotion tags, aftercare | "Every kept object gets an identity" | 「每个被保存的对象都有身份」 |
| 6 | **Membership** | `/membership` Core Sequence plans (Lite/Pro quotas and benefits) | "Core Sequence — quotas that scale" | 「核心序列会员 · 额度随你生长」 |

Feature graphic (1024 × 500): ink background (`#0a0c11`), gold meridian line,
wordmark ZOTAIX left, a bottle-and-certificate composition right, tagline
"Emotions, made into objects".

## Capture workflow

1. Run the deployed site (or `npm run dev`) with seeded data — the shot list
   above maps to seeded records, so screens are reproducible.
2. Capture through the wrapper or a device simulator at exact pixel sizes:
   - iOS: Xcode Simulator (iPhone 15 Pro Max → 1290 × 2796; iPad Pro 12.9");
     `Cmd+S` saves at native resolution.
   - Android: Pixel emulator at 1080 × 1920, or Chrome DevTools device mode
     with DPR set to 3.
3. Dark theme only — the platform is designed on ink/obsidian; keep the
   status bar clean (full battery, no notifications, 9:41 on iOS).
4. Produce one EN set and one ZH set (switch the locale before capturing);
   upload per-locale in both consoles.
5. Frame with captions above the device render; caption text color
   porcelain `#f5f2ea` on ink, accent gold for one keyword; keep captions
   ≤ 5 words (EN) / ≤ 10 chars (ZH).
6. No alcohol-consumption imagery anywhere — bottles, certificates, and UI
   only. This keeps screenshots inside every regional advertising rule the
   platform already follows.

## File naming

```
screenshots/
  ios/6.7/en/01-concierge.png ... 06-membership.png
  ios/6.7/zh/01-concierge.png ...
  ios/6.5/... ios/5.5/... ios/ipad-12.9/...
  android/phone/en/01-concierge.png ...
  android/feature-graphic/feature-1024x500.png
```
