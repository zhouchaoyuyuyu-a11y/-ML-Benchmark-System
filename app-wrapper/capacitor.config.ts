// ZOTAIX app wrapper — Capacitor configuration.
//
// This directory is intentionally OUTSIDE the Next.js tsconfig (see repo
// tsconfig.json "exclude"); it is its own small npm project. Install deps
// here before running Capacitor commands:
//
//   cd app-wrapper
//   npm init -y
//   npm i @capacitor/core @capacitor/cli @capacitor/ios @capacitor/android
//   npx cap add ios && npx cap add android
//
// Strategy: REMOTE-URL WRAPPER. The Next.js platform is dynamic (server
// components, API routes, session cookies), so the native shell loads the
// deployed site directly via `server.url` instead of bundling a static
// export. See ../docs/APP_PACKAGING.md for the full rationale and the
// offline-splash setup.

import type { CapacitorConfig } from "@capacitor/cli";

// Production wrapper mode: point at the deployed site. Configure via the
// ZOTAIX_SITE_URL environment variable at build time; the default matches
// NEXT_PUBLIC_SITE_URL's fallback in src/lib/config.ts. For device testing
// against a local dev server, export ZOTAIX_SITE_URL=http://<lan-ip>:3000
// (and only then set server.cleartext to true).
const SITE_URL = process.env.ZOTAIX_SITE_URL || "https://zotaix-web.vercel.app";

const config: CapacitorConfig = {
  appId: "com.zotaix.app",
  appName: "ZOTAIX",

  // webDir is required by the Capacitor CLI even in remote-URL mode. It holds
  // only the local offline splash (www/index.html) that shows when the
  // device has no connectivity; every online session is served from
  // server.url. Copy icons from ../public/icons when generating native
  // assets (see README.md in this directory).
  webDir: "www",

  server: {
    // Remote-URL wrapper mode: the WebView loads the live platform, so web
    // deploys ship to the app instantly and session cookies stay first-party
    // on the site domain (persistent login).
    url: SITE_URL,
    cleartext: false,
    // Keep navigation inside the platform; external links (social matrix,
    // store pages) open in the system browser.
    allowNavigation: ["zotaix-web.vercel.app"],
  },

  ios: {
    // Custom scheme for the WebView origin; deep links use Universal Links
    // (https) plus the zotaix:// scheme registered in Info.plist. Routes
    // handled: /reserve, /co-create, /concierge — see docs/APP_PACKAGING.md.
    scheme: "zotaix",
    contentInset: "automatic",
    backgroundColor: "#0a0c11",
    limitsNavigationsToAppBoundDomains: false,
  },

  android: {
    // App Links (https, verified via /.well-known/assetlinks.json) plus the
    // zotaix:// scheme registered in AndroidManifest.xml map to the same
    // routes: /reserve, /co-create, /concierge.
    backgroundColor: "#0a0c11",
    allowMixedContent: false,
  },
};

export default config;
