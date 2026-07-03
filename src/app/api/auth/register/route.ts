import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { encodeSession, registerUser, SESSION_COOKIE_NAME } from "@/lib/auth";

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));
  const result = registerUser(String(body.email ?? ""), String(body.password ?? ""), String(body.nickname ?? ""));
  if (!result.ok || !result.user) {
    return NextResponse.json({ ok: false, error: result.error }, { status: 400 });
  }
  const store = await cookies();
  store.set(SESSION_COOKIE_NAME, encodeSession(result.user.id), {
    path: "/",
    httpOnly: true,
    sameSite: "lax",
    maxAge: 60 * 60 * 24 * 30,
  });
  return NextResponse.json({ ok: true, user: { id: result.user.id, nickname: result.user.nickname, user_type: result.user.user_type } });
}
