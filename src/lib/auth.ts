import { createHmac, createHash } from "crypto";
import { cookies } from "next/headers";
import { sessionSecret } from "./config";
import { db, newId, now } from "./store";
import type { User } from "./types";

const SESSION_COOKIE = "zx_session";
const VISITOR_COOKIE = "zx_visitor";

function sign(value: string): string {
  return createHmac("sha256", sessionSecret).update(value).digest("hex").slice(0, 32);
}

export function hashPassword(password: string): string {
  return createHash("sha256").update(`zx:${password}`).digest("hex");
}

export function encodeSession(userId: string): string {
  const payload = Buffer.from(JSON.stringify({ uid: userId })).toString("base64url");
  return `${payload}.${sign(payload)}`;
}

export function decodeSession(token?: string): string | null {
  if (!token) return null;
  const [payload, sig] = token.split(".");
  if (!payload || !sig || sign(payload) !== sig) return null;
  try {
    const parsed = JSON.parse(Buffer.from(payload, "base64url").toString());
    return typeof parsed.uid === "string" ? parsed.uid : null;
  } catch {
    return null;
  }
}

/** Current signed-in user, or null for guests. */
export async function getSessionUser(): Promise<User | null> {
  const store = await cookies();
  const uid = decodeSession(store.get(SESSION_COOKIE)?.value);
  if (!uid) return null;
  return db().users.find((u) => u.id === uid) ?? null;
}

/** Anonymous visitor id (for guest quota); empty string when cookie unset. */
export async function getVisitorId(): Promise<string> {
  const store = await cookies();
  return store.get(VISITOR_COOKIE)?.value ?? "";
}

export const SESSION_COOKIE_NAME = SESSION_COOKIE;
export const VISITOR_COOKIE_NAME = VISITOR_COOKIE;

export interface AuthResult {
  ok: boolean;
  user?: User;
  error?: string;
}

export function registerUser(email: string, password: string, nickname: string): AuthResult {
  const cleanEmail = email.trim().toLowerCase();
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(cleanEmail)) return { ok: false, error: "Please enter a valid email address." };
  if (password.length < 6) return { ok: false, error: "Password must be at least 6 characters." };
  const database = db();
  if (database.users.some((u) => u.email === cleanEmail)) return { ok: false, error: "An account with this email already exists — sign in instead." };
  const user: User = {
    id: newId("usr"),
    email: cleanEmail,
    password_hash: hashPassword(password),
    nickname: nickname.trim() || cleanEmail.split("@")[0],
    user_type: "registered",
    membership_level: "free",
    daily_quota: database.settings.free_daily_chat,
    used_quota: 0,
    created_at: now(),
    updated_at: now(),
  };
  database.users.push(user);
  return { ok: true, user };
}

export function loginUser(email: string, password: string): AuthResult {
  const cleanEmail = email.trim().toLowerCase();
  const user = db().users.find((u) => u.email === cleanEmail);
  if (!user) return { ok: false, error: "No account found for this email." };
  // Seeded demo accounts accept the documented demo password.
  const demoOk = !user.password_hash && password === "zotaix-demo";
  const hashOk = !!user.password_hash && user.password_hash === hashPassword(password);
  if (!demoOk && !hashOk) return { ok: false, error: "Incorrect password." };
  return { ok: true, user };
}
