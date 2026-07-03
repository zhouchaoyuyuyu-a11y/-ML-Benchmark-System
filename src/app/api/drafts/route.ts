import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { db, newId, now, versionHash } from "@/lib/store";
import type { ConceptProposal, ObjectType } from "@/lib/types";

/** Save an AI proposal (or manual form) as an object draft. */
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) {
    return NextResponse.json(
      { ok: false, error: "Saving to your archive requires a free account." },
      { status: 401 }
    );
  }
  const body = await req.json().catch(() => ({}));
  const proposal: ConceptProposal | undefined = body.proposal;
  const objectTypes: ObjectType[] = ["spirit", "fragrance", "bottle", "giftbox", "label", "enterprise_gift"];
  const objectType: ObjectType = objectTypes.includes(body.object_type) ? body.object_type : proposal?.scent_direction && !proposal?.liquid_direction ? "fragrance" : "spirit";

  const database = db();
  const draft = {
    id: newId("dft"),
    user_id: user.id,
    object_type: objectType,
    title: String(body.title ?? proposal?.names?.[0] ?? "Untitled inspiration").slice(0, 120),
    scene: body.scene ? String(body.scene).slice(0, 200) : undefined,
    recipient: body.recipient ? String(body.recipient).slice(0, 120) : undefined,
    budget: body.budget ? String(body.budget).slice(0, 60) : undefined,
    emotion_tags: Array.isArray(body.emotion_tags) ? body.emotion_tags.slice(0, 6).map(String) : proposal?.keywords ?? [],
    liquid_direction: proposal?.liquid_direction ?? (body.liquid_direction ? String(body.liquid_direction) : undefined),
    scent_direction: proposal?.scent_direction ?? (body.scent_direction ? String(body.scent_direction) : undefined),
    label_copy: proposal?.label_copy ?? (body.label_copy ? String(body.label_copy) : undefined),
    visual_style: proposal?.bottle_direction ?? (body.visual_style ? String(body.visual_style) : undefined),
    names: proposal?.names,
    status: "saved" as const,
    public_visible: Boolean(body.public_visible ?? false),
    created_at: now(),
    updated_at: now(),
  };
  database.object_drafts.push(draft);

  // First design version snapshot for the Design module.
  database.design_versions.push({
    id: newId("ver"),
    object_draft_id: draft.id,
    version_name: "v1 · Initial direction",
    design_payload: {
      bottle: draft.visual_style ?? "To be defined in Studio",
      label: draft.label_copy ?? "To be defined in Design",
      liquid: draft.liquid_direction ?? "—",
      scent: draft.scent_direction ?? "—",
    },
    version_hash: versionHash(draft),
    created_at: now(),
  });

  return NextResponse.json({ ok: true, draft });
}

/** List the signed-in user's drafts. */
export async function GET() {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ ok: false, error: "Sign in required." }, { status: 401 });
  const drafts = db()
    .object_drafts.filter((d) => d.user_id === user.id)
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  return NextResponse.json({ ok: true, drafts });
}
