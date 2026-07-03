# ZOTAIX app wrapper (Capacitor)

Native iOS/Android shells around the deployed ZOTAIX web platform. This
directory is excluded from the Next.js tsconfig — it is an independent small
project. Full strategy, deep-link JSON samples, and the store submission
checklist live in [`../docs/APP_PACKAGING.md`](../docs/APP_PACKAGING.md);
this README is the operational quickstart.

## Why a remote-URL wrapper

The platform is a dynamic Next.js app (server components, `/api/*` routes,
session cookies), so a static export cannot represent it. The wrapper's
WebView loads the deployed site (`server.url` in `capacitor.config.ts`):

- Web deploys reach the app instantly — no store re-review for content.
- Login persists: the session cookie is first-party on the site domain
  because the WebView loads that domain directly.
- The only local asset is an offline splash in `www/` shown without
  connectivity.

The PWA remains the first-class install path (`/download`); the wrapper adds
store presence, deep-link handling, and a home-screen identity on devices
where PWA install is weak (notably WebView-in-WeChat — see caveats in
`docs/APP_PACKAGING.md`).

## Setup

```bash
cd app-wrapper
npm init -y
npm i @capacitor/core @capacitor/cli @capacitor/ios @capacitor/android
mkdir -p www   # offline splash lives here (see below)

# capacitor.config.ts is already in this directory — `npx cap init` is not
# needed unless you are recreating the config from scratch.
npx cap add ios
npx cap add android
npx cap sync
```

Point the shell at your deployment (defaults to the production site):

```bash
export ZOTAIX_SITE_URL="https://zotaix-web.vercel.app"
npx cap sync
```

## Offline splash (`www/index.html`)

Create a single self-contained page: ink background `#0a0c11`, the ZOTAIX
wordmark, porcelain text "ZOTAIX opens when you're back online / 网络恢复后即刻进入",
and a retry button that reloads `window.location`. No external requests. The
WebView shows it only when the remote site is unreachable.

## Icons & splash screens

Source assets are already in the repo:

- `public/icons/icon-512.png`, `icon-192.png`, `icon-maskable-512.png`,
  `apple-touch-icon.png`
- Store-size exports and the 1024×500 feature graphic per
  `app-store-assets/screenshots/README.md`

Generate native sets with:

```bash
npm i -D @capacitor/assets
mkdir -p assets
cp ../public/icons/icon-512.png assets/icon.png        # regenerate a 1024px master if needed
npx capacitor-assets generate --iconBackgroundColor '#0a0c11' --splashBackgroundColor '#0a0c11'
```

## iOS

```bash
npx cap open ios
```

- Bundle ID `com.zotaix.app`; set your Team for signing.
- Version = `app_config.latest_version` (e.g. `1.4.0`); build number
  increments per upload.
- Associated Domains capability: `applinks:zotaix-web.vercel.app` (Universal
  Links; the site serves `/.well-known/apple-app-site-association` — sample
  JSON in `docs/APP_PACKAGING.md`).
- Register the `zotaix://` URL scheme in Info.plist (`CFBundleURLTypes`).
- Push: the wrapper ships without push registration — the platform notifies
  through in-app Reserve/co-creation status and email. If push is adopted,
  add the Push Notifications capability and an APNs key per
  `docs/APP_PACKAGING.md`.

## Android

```bash
npx cap open android
```

- Package `com.zotaix.app`; `versionName` = `app_config.latest_version`,
  `versionCode` per the mapping in `app-store-assets/changelog-template.md`
  (1.4.0 → 10400).
- Create and guard a release keystore (`keytool -genkeypair ...`); losing it
  forfeits the package name on Play. Prefer Play App Signing.
- App Links: intent filter with `android:autoVerify="true"` for
  `https://zotaix-web.vercel.app` paths `/reserve`, `/co-create`,
  `/concierge`; the site serves `/.well-known/assetlinks.json` — sample in
  `docs/APP_PACKAGING.md`.
- Also register the `zotaix://` scheme intent filter.
- Build: `./gradlew bundleRelease` (Play `.aab`) and `./gradlew
  assembleRelease` (direct-distribution APK for `APP_APK_URL`).

## Deep links handled

| Link | Opens |
| --- | --- |
| `zotaix://reserve` / `https://<site>/reserve...` | Reserve archive & certificates |
| `zotaix://co-create` / `https://<site>/co-create...` | Co-creation pool |
| `zotaix://concierge` / `https://<site>/concierge...` | AI concierge |

In remote-URL mode https links need no JS routing — the WebView is already on
the site, so it navigates natively. The `zotaix://` scheme is translated to
the matching path via the `appUrlOpen` listener (snippet in
`docs/APP_PACKAGING.md`).

## Versioning & forced update

The site reads `app_config` (`latest_version`, `force_update_version`).
Wrappers report their version via the `zx-app-version` query/user-agent hook;
when a wrapper is older than `force_update_version`, the site presents the
update screen with the store links from `APP_IOS_URL`/`APP_ANDROID_URL`.
Keep native version numbers in lockstep with `app_config.latest_version`.

## Store submission

Follow the checklist in `docs/APP_PACKAGING.md`, using the listing copy in
`app-store-assets/` (name/subtitle, descriptions, keywords, screenshots,
privacy answers, release notes).
