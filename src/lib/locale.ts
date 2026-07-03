import { cookies } from "next/headers";
import type { Locale } from "./types";

export const LOCALE_COOKIE = "zx_lang";

export async function getLocale(): Promise<Locale> {
  const store = await cookies();
  const v = store.get(LOCALE_COOKIE)?.value;
  return v === "zh" ? "zh" : "en";
}

/** Bilingual helper: pick the string for the active locale. */
export function tt(locale: Locale, en: string, zh: string): string {
  return locale === "zh" ? zh : en;
}
