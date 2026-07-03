# Changelog / release notes template

One source of truth: `app_config.changelog` (admin panel → App config). Store
release notes, the in-app changelog on `/download`, and git tags are all
written from the same entry.

## Versioning (semver)

- **MAJOR** — navigation or account-model changes that require relearning.
- **MINOR** — new user-facing capability (a new concierge mode, a new line).
- **PATCH** — fixes and copy/performance improvements; usually not a store
  release on its own.

Wrapper mapping (see `docs/APP_PACKAGING.md`):

- iOS `CFBundleShortVersionString` / Android `versionName` = `latest_version`
- Android `versionCode` = `MAJOR*10000 + MINOR*100 + PATCH` (1.4.0 → 10400)
- `app_config.force_update_version` marks the oldest version that may keep
  running; wrappers older than it are shown the update screen by the site.

## Entry template

```json
{
  "version": "X.Y.Z",
  "date": "YYYY-MM-DD",
  "notes": [
    "User-visible change, stated as what it does",
    "Second change",
    "Third change"
  ]
}
```

Writing rules for `notes`:

- Lead with the noun users know ("Reserve certificates…", "Co-creation…").
- Present tense, no internal jargon, no issue numbers.
- 1–4 items; if it needs more, it should have been two releases.
- Same items go verbatim into App Store "What's New" and Play release notes
  (both accept plain text; prefix each with "• " there).

## Worked example — current shipped history (matches the app_config seed)

```json
[
  {
    "version": "1.4.0",
    "date": "2026-06-20",
    "notes": [
      "Reserve certificates with QR binding",
      "Co-creation pool voting",
      "WeChat sharing cards"
    ]
  },
  {
    "version": "1.3.0",
    "date": "2026-05-12",
    "notes": [
      "Studio bottle preview",
      "Membership quota center"
    ]
  },
  {
    "version": "1.2.0",
    "date": "2026-04-02",
    "notes": [
      "AI concierge structured proposals",
      "Emotional cards"
    ]
  }
]
```

Store rendering of 1.4.0:

```
• Reserve certificates with QR binding
• Co-creation pool voting
• WeChat sharing cards
```

中文商店文案（zh-Hans 发布说明，与英文条目一一对应）：

```
• 档案证书支持二维码绑定
• 共创池投票
• 微信分享卡片
```

## Release checklist

1. Add the new entry to `app_config.changelog` and bump
   `app_config.latest_version` in the admin panel.
2. If old wrappers must upgrade (breaking web/API change), raise
   `force_update_version`.
3. Update wrapper version numbers (`versionName`/`versionCode`,
   `CFBundleShortVersionString`/`CFBundleVersion`) to match.
4. Paste the notes into App Store Connect "What's New" (EN + ZH) and Play
   Console release notes (en-US + zh-CN).
5. Tag the repo: `git tag app-vX.Y.Z`.
