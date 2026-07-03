import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { db, newId, newZotaixId, now, versionHash } from "@/lib/store";

/** Turn an object draft into a Reserve record (digital archive entry). */
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ ok: false, error: "Sign in required." }, { status: 401 });
  const body = await req.json().catch(() => ({}));
  const database = db();
  const draft = database.object_drafts.find((d) => d.id === body.draftId && d.user_id === user.id);
  if (!draft) return NextResponse.json({ ok: false, error: "Draft not found." }, { status: 404 });

  const isPrivateAllowed = user.membership_level !== "free";
  const requestedPrivacy = body.privacy_level === "private" ? "private" : "public";
  const record = {
    id: newId("rsv"),
    user_id: user.id,
    object_draft_id: draft.id,
    zotaix_id: newZotaixId(),
    object_type: draft.object_type,
    object_name: draft.title,
    emotion_tags: draft.emotion_tags,
    relationship_scene: [draft.recipient, draft.scene].filter(Boolean).join(" · ") || undefined,
    product_direction: draft.liquid_direction ?? draft.scent_direction,
    label_copy: draft.label_copy,
    scent_direction: draft.scent_direction,
    liquid_direction: draft.liquid_direction,
    visual_style: draft.visual_style,
    qr_nfc_id: `QR-${versionHash(draft.id + now()).toUpperCase().replace("ZX-", "ZX-")}`,
    privacy_level: (isPrivateAllowed ? requestedPrivacy : "public") as "public" | "private",
    co_create_eligible: draft.public_visible,
    delivery_status: "digital" as const,
    repurchase_eligible: draft.object_type !== "label",
    aftercare_status: "none" as const,
    created_at: now(),
    updated_at: now(),
  };
  database.reserve_records.push(record);
  draft.status = "reviewed";
  draft.updated_at = now();

  return NextResponse.json({
    ok: true,
    record,
    note:
      !isPrivateAllowed && requestedPrivacy === "private"
        ? "Private Reserve records are a Core Sequence benefit — this record was archived as public."
        : undefined,
  });
}
