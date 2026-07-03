import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { LOCALE_COOKIE } from "@/lib/locale";

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));
  const locale = body?.locale === "zh" ? "zh" : "en";
  const store = await cookies();
  store.set(LOCALE_COOKIE, locale, { path: "/", maxAge: 60 * 60 * 24 * 365, sameSite: "lax" });
  return NextResponse.json({ ok: true, locale });
}
