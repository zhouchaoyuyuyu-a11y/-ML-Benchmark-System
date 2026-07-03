import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { db, newId, now } from "@/lib/store";
import type { UserProfile } from "@/lib/types";

const LIST_FIELDS = ["favorite_colors", "scent_preferences", "alcohol_preferences", "visual_preferences", "gift_preferences", "common_scenarios", "personality_tags"] as const;
const TEXT_FIELDS = ["mbti", "zodiac", "blood_type", "age_range", "nickname", "address_style", "alcohol_tolerance", "music", "movies", "cities", "literary_imagery", "budget_range", "emotional_state"] as const;

/** Read or update the signed-in user's preference profile. */
export async function GET() {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ ok: false, error: "Sign in required." }, { status: 401 });
  const profile = db().user_profiles.find((p) => p.user_id === user.id) ?? null;
  const relationships = db().relationship_profiles.filter((r) => r.user_id === user.id);
  return NextResponse.json({ ok: true, profile, relationships });
}

export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ ok: false, error: "Sign in required." }, { status: 401 });
  const body = await req.json().catch(() => ({}));
  const database = db();

  if (body.action === "delete") {
    database.user_profiles = database.user_profiles.filter((p) => p.user_id !== user.id);
    return NextResponse.json({ ok: true, deleted: true });
  }

  let profile = database.user_profiles.find((p) => p.user_id === user.id);
  if (!profile) {
    profile = {
      id: newId("prf"),
      user_id: user.id,
      privacy_level: "private",
      memory_enabled: true,
      created_at: now(),
      updated_at: now(),
    } as UserProfile;
    database.user_profiles.push(profile);
  }

  for (const f of TEXT_FIELDS) {
    if (typeof body[f] === "string") profile[f] = body[f].slice(0, 300) || undefined;
  }
  for (const f of LIST_FIELDS) {
    if (Array.isArray(body[f])) profile[f] = body[f].slice(0, 12).map((v: unknown) => String(v).slice(0, 60));
  }
  if (typeof body.non_alcohol_ok === "boolean") profile.non_alcohol_ok = body.non_alcohol_ok;
  if (typeof body.memory_enabled === "boolean") profile.memory_enabled = body.memory_enabled;
  if (["private", "co-create", "public"].includes(body.privacy_level)) profile.privacy_level = body.privacy_level;
  profile.updated_at = now();

  if (body.relationship && typeof body.relationship === "object") {
    const r = body.relationship;
    database.relationship_profiles.push({
      id: newId("rel"),
      user_id: user.id,
      relation_type: String(r.relation_type ?? "Friend").slice(0, 60),
      nickname: String(r.nickname ?? "—").slice(0, 60),
      age_range: r.age_range ? String(r.age_range).slice(0, 30) : undefined,
      preferences: r.preferences ? String(r.preferences).slice(0, 400) : undefined,
      important_dates: r.important_dates ? String(r.important_dates).slice(0, 200) : undefined,
      notes: r.notes ? String(r.notes).slice(0, 400) : undefined,
      privacy_level: "private",
      created_at: now(),
      updated_at: now(),
    });
  }

  return NextResponse.json({ ok: true, profile });
}
