# ZOTAIX — store listing assets

Everything needed to fill in App Store Connect and Google Play Console for the
ZOTAIX app (the Capacitor wrapper defined in `app-wrapper/`, wrapping the
deployed web platform).

## Contents

| File | Purpose |
| --- | --- |
| `app-name.md` | App name, subtitles (EN/ZH), categories, age rating guidance |
| `descriptions.md` | Long descriptions (App Store + Google Play, EN/ZH), short description, promotional text |
| `keywords.md` | iOS 100-character keyword strings (EN/ZH) + Google Play keyword strategy |
| `screenshots/README.md` | Required screenshot sizes and the canonical shot list |
| `changelog-template.md` | Semver release-note template + worked example matching the in-app changelog |
| `privacy-notes.md` | App Store privacy nutrition labels + Google Play data safety mapping, alcohol disclosure, account deletion |

## How these map to the platform

- The app is a wrapper around the live site (`NEXT_PUBLIC_SITE_URL`), so store
  copy describes exactly what the web platform does today: AI-generated
  creative proposals, human-confirmed physical delivery, the Reserve archive,
  the co-creation pool, and Core Sequence membership.
- Version numbers in store releases must match `app_config.latest_version`
  (admin panel → App config). The in-app changelog (`app_config.changelog`)
  and the store release notes are written from the same template
  (`changelog-template.md`).
- Once a listing is live, paste the store URLs into `APP_IOS_URL` /
  `APP_ANDROID_URL` / `APP_APK_URL` (see `.env.example`) or set them in the
  admin app-config panel; the `/download` page picks them up immediately.

## Submission flow

1. Build and sign the wrapper — see `docs/APP_PACKAGING.md`.
2. Fill listing metadata from `app-name.md`, `descriptions.md`, `keywords.md`.
3. Upload screenshots per `screenshots/README.md`.
4. Complete privacy questionnaires from `privacy-notes.md`.
5. Set age rating per `app-name.md` (alcohol references — see regional notes).
6. Write release notes from `changelog-template.md`, keeping
   `app_config.changelog` in sync.
