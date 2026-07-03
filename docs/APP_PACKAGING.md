# ZOTAIX — app packaging guide

How the web platform becomes an installable app: PWA first, then native
iOS/Android shells via the Capacitor wrapper in `app-wrapper/`. Store listing
copy and questionnaires live in `app-store-assets/`.

---

## 1. Strategy: PWA first, wrapper for store presence

**PWA is the primary install path.** The platform already ships
`public/manifest.webmanifest`, `public/sw.js`, and the icon set in
`public/icons/`; `/download` walks users through installation, and
`app_config.pwa_enabled` / `install_prompt_enabled` control the banner and
prompt. A PWA install delivers the full product — every feature is the live
site.

**The native wrapper adds what a PWA cannot:**

- Presence in the App Store and Google Play (discovery + trust).
- Verified deep links (Universal Links / App Links) into `/reserve`,
  `/co-create`, `/concierge`.
- A reliable home-screen identity on devices and browsers where PWA install
  is weak — most importantly inside WeChat's WebView (section 9).

## 2. Wrapper architecture: remote URL, not static export

Two ways to wrap a Next.js app in Capacitor:

| | Remote-URL wrapper (chosen) | Static export bundle |
| --- | --- | --- |
| How | WebView loads the deployed site via `server.url` | `next export` output copied into the app |
| Dynamic Next app (server components, `/api/*`, sessions) | Works — it IS the site | Not possible: this app depends on server rendering and API routes |
| Content updates | Instant with each web deploy; no store re-review | Requires a store release per change |
| Offline | Offline splash page only | Static shell offline, but data calls still need network |
| Session cookies | First-party on the site domain — login persists | Cookies cross-origin from `capacitor://` — fragile |
| Store review risk | Fine when the app is a real product experience (it is); keep the native shell adding deep links, icons, splash | Lower novelty, higher maintenance |

**Decision: remote-URL wrapper pointing at the deployed Vercel site, with a
local offline splash.** `app-wrapper/capacitor.config.ts` already encodes
this (`server.url` from `ZOTAIX_SITE_URL`, default
`https://zotaix-web.vercel.app`).

## 3. Wrapper setup (commands)

```bash
cd app-wrapper
npm init -y
npm i @capacitor/core @capacitor/cli @capacitor/ios @capacitor/android

# capacitor.config.ts ships in this repo. If recreating from scratch:
#   npx cap init "ZOTAIX" "com.zotaix.app" --web-dir www

mkdir -p www            # offline splash (self-contained index.html)
npx cap add ios
npx cap add android
npx cap sync
```

`webDir: "www"` exists to satisfy the CLI and hold the offline splash: a
single self-contained `index.html` (ink `#0a0c11` background, ZOTAIX
wordmark, "ZOTAIX opens when you're back online / 网络恢复后即刻进入", retry
button). Online sessions never touch it.

Build against a specific deployment:

```bash
ZOTAIX_SITE_URL="https://zotaix-web.vercel.app" npx cap sync
```

## 4. iOS configuration

1. **Bundle ID** `com.zotaix.app` (App Store Connect → new app with the same
   ID). Display name `ZOTAIX`.
2. **Signing:** Xcode → Signing & Capabilities → select the team; use
   automatically managed signing for development, a distribution certificate
   + App Store provisioning profile for release.
3. **Versioning:** `CFBundleShortVersionString` = `app_config.latest_version`
   (`1.4.0`); `CFBundleVersion` increments every upload.
4. **Universal Links:** add the Associated Domains capability with
   `applinks:zotaix-web.vercel.app`. The site must serve
   `https://<site>/.well-known/apple-app-site-association` (content type
   `application/json`, no redirect). Sample:

```json
{
  "applinks": {
    "apps": [],
    "details": [
      {
        "appIDs": ["TEAMID1234.com.zotaix.app"],
        "components": [
          { "/": "/reserve/*", "comment": "Reserve certificates (QR scans land here)" },
          { "/": "/reserve", "comment": "Reserve archive" },
          { "/": "/co-create/*", "comment": "Co-creation project pages" },
          { "/": "/co-create", "comment": "Co-creation pool" },
          { "/": "/concierge", "comment": "AI concierge" }
        ]
      }
    ]
  },
  "webcredentials": { "apps": ["TEAMID1234.com.zotaix.app"] }
}
```

   Replace `TEAMID1234` with the real Team ID. Serve it from
   `public/.well-known/apple-app-site-association` on the Next site.
5. **Custom scheme:** register `zotaix://` in Info.plist
   (`CFBundleURLTypes` → `CFBundleURLSchemes: ["zotaix"]`).
6. **Push notifications:** the wrapper ships without push registration; the
   product communicates through in-app Reserve/co-creation status and email,
   so no APNs entitlement is needed for submission. If push is adopted,
   enable the Push Notifications capability, create an APNs key in the
   developer portal, and add `@capacitor/push-notifications` — the listing
   privacy answers in `app-store-assets/privacy-notes.md` must then be
   updated to declare the device token.

## 5. Android configuration

1. **Package name** `com.zotaix.app` (immutable on Play after first upload).
2. **Keystore:**

```bash
keytool -genkeypair -v -keystore zotaix-release.keystore \
  -alias zotaix -keyalg RSA -keysize 4096 -validity 10000
```

   Store it outside the repo; enroll in Play App Signing so Google holds the
   final signing key and the local keystore is the upload key.
3. **Versioning:** `versionName` = `app_config.latest_version`;
   `versionCode` = `MAJOR*10000 + MINOR*100 + PATCH` (1.4.0 → `10400`). The
   monotonic code is what Play enforces.
4. **App Links:** intent filter in `AndroidManifest.xml` with
   `android:autoVerify="true"` for host `zotaix-web.vercel.app`, path
   prefixes `/reserve`, `/co-create`, `/concierge`; plus a second intent
   filter for the `zotaix` scheme. The site must serve
   `https://<site>/.well-known/assetlinks.json`:

```json
[
  {
    "relation": ["delegate_permission/common.handle_all_urls"],
    "target": {
      "namespace": "android_app",
      "package_name": "com.zotaix.app",
      "sha256_cert_fingerprints": [
        "AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99"
      ]
    }
  }
]
```

   Use the SHA-256 from Play App Signing (Console → Setup → App integrity),
   not the upload key. Serve from `public/.well-known/assetlinks.json`.
5. **Builds:** `./gradlew bundleRelease` → `.aab` for Play;
   `./gradlew assembleRelease` → signed APK for direct distribution
   (`APP_APK_URL`, used for regions without Play access).

## 6. Icons & splash

Masters live in `public/icons/` (`icon-512.png`, `icon-192.png`,
`icon-maskable-512.png`, `apple-touch-icon.png`). Generate native sets:

```bash
cd app-wrapper
npm i -D @capacitor/assets
mkdir -p assets && cp ../public/icons/icon-512.png assets/icon.png
npx capacitor-assets generate \
  --iconBackgroundColor '#0a0c11' --splashBackgroundColor '#0a0c11'
```

Store-side art (1024×1024 App Store icon, 512×512 Play icon, 1024×500
feature graphic, screenshots) follows
`app-store-assets/screenshots/README.md`.

## 7. Sessions, login persistence, cookies

- Authentication is a signed session cookie set by the Next site
  (`SESSION_SECRET`). In remote-URL mode the WebView's page origin **is**
  the site domain, so the cookie is first-party: login survives app
  restarts, and no token bridge is needed.
- Keep `server.url` on the exact canonical domain (`NEXT_PUBLIC_SITE_URL`);
  a `www.`/apex mismatch would scope cookies to the wrong host.
- If the site ever moves behind an auth-scoped subdomain, set the cookie
  `Domain` attribute to the parent domain so wrapper and browser share it.
- iOS ships cookies through `WKWebView`'s persistent store by default;
  Android WebView likewise — no extra plugin required.

## 8. Deep links → routes

Targets and sources:

| Route | https (Universal/App Links) | Custom scheme |
| --- | --- | --- |
| `/reserve`, `/reserve/[id]` | QR codes on certificates and bottles | `zotaix://reserve` |
| `/co-create`, `/co-create/[id]` | shared project pages, WeChat menu | `zotaix://co-create` |
| `/concierge` | marketing links, WeChat auto-replies | `zotaix://concierge` |

https links need no handling in remote-URL mode — the OS opens the app and
the WebView is already on that origin. The custom scheme is translated with
the App plugin (add `@capacitor/app` and this to the wrapper's tiny JS in
`www/`, or a native equivalent):

```ts
import { App } from "@capacitor/app";

App.addListener("appUrlOpen", ({ url }) => {
  const u = new URL(url);
  if (u.protocol === "zotaix:") {
    // zotaix://reserve/rsv_001 → https://<site>/reserve/rsv_001
    const path = `/${u.host}${u.pathname}`.replace(/\/+$/, "");
    window.location.href = `${SITE_URL}${path || "/"}`;
  }
});
```

## 9. WebView-in-WeChat caveats

A large share of Chinese traffic arrives inside WeChat's built-in browser,
which is not the system browser:

- **No PWA install and no store handoff:** WeChat blocks `beforeinstallprompt`
  and intercepts App Store / Play links. The `/download` page detects the
  WeChat UA and shows the "open in system browser" overlay (the ⋯ menu →
  "Open in Browser") — this is designed behavior, keep it.
- **APK downloads are blocked** in the WeChat WebView; the APK path
  (`APP_APK_URL`) must also go through the system-browser overlay.
- **Universal Links/App Links do not fire** from inside WeChat conversations
  reliably; links opened in WeChat stay in its WebView. The platform
  therefore keeps every feature usable in plain web form — the WeChat menu
  (`wechat_config.menu_config`) targets site routes, not app links.
- **Session isolation:** WeChat's WebView has its own cookie jar; users may
  need to sign in there separately. Harmless, but support should know.
- **Payments:** inside WeChat, WeChat Pay (JSAPI) is the only in-context
  payment; until `WECHAT_PAY_*` credentials are configured, orders route to
  human concierge confirmation — the designed fallback.

## 10. Force-update flow (app_config)

`app_config` is the control surface (admin panel → App config):

- `latest_version` — what the newest wrapper is (also drives `/download`).
- `force_update_version` — the oldest wrapper version allowed to keep
  running. Seeded example: latest `1.4.0`, force `1.2.0` → a `1.1.x` wrapper
  is shown a blocking update screen with the store buttons
  (`APP_IOS_URL`/`APP_ANDROID_URL`/`APP_APK_URL`); `1.2.0+` keeps working.
- Raise `force_update_version` only for breaking web/API changes; routine
  releases just bump `latest_version`.
- `changelog` feeds `/download` and store release notes — one source of
  truth, template in `app-store-assets/changelog-template.md`.

## 11. Store submission checklist

1. **Deploy** the site; verify `/.well-known/apple-app-site-association` and
   `/.well-known/assetlinks.json` return JSON over HTTPS with no redirect.
2. **Versions:** bump `app_config.latest_version` + changelog entry; set
   matching `versionName`/`versionCode` and
   `CFBundleShortVersionString`/`CFBundleVersion`.
3. **Build:** `npx cap sync`, then Xcode archive → App Store Connect upload;
   `./gradlew bundleRelease` → Play Console upload.
4. **Listing copy:** name/subtitle from `app-store-assets/app-name.md`;
   descriptions and promotional text from `app-store-assets/descriptions.md`;
   iOS keyword strings from `app-store-assets/keywords.md`.
5. **Screenshots:** per `app-store-assets/screenshots/README.md` (EN + ZH
   sets, six canonical shots, feature graphic for Play).
6. **Age rating:** answers from `app-store-assets/app-name.md` (Apple 17+,
   IARC alcohol references) + the review note about human-confirmed delivery
   and the 18+ age gate.
7. **Privacy:** nutrition labels / Data safety from
   `app-store-assets/privacy-notes.md`; privacy policy URL
   `/legal/privacy`; account deletion URL `/profile`.
8. **Release notes:** from `app-store-assets/changelog-template.md`, EN + ZH.
9. **After approval:** paste store URLs into `APP_IOS_URL` /
   `APP_ANDROID_URL` (env or admin app-config) — `/download` and the
   force-update screen pick them up immediately.
10. **Tag:** `git tag app-vX.Y.Z && git push --tags`.
