import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { db, newId, now } from "@/lib/store";

/** Create a co-creation project (members only) or join/vote on one. */
export async function POST(req: Request) {
  const user = await getSessionUser();
  const body = await req.json().catch(() => ({}));
  const database = db();
  const action = String(body.action ?? "create");

  if (action === "vote") {
    const project = database.co_creation_projects.find((p) => p.id === body.projectId && p.public_visible);
    if (!project) return NextResponse.json({ ok: false, error: "Project not found." }, { status: 404 });
    project.votes += 1;
    project.updated_at = now();
    return NextResponse.json({ ok: true, votes: project.votes });
  }

  if (!user) {
    return NextResponse.json({ ok: false, error: "Joining or starting co-creation requires an account." }, { status: 401 });
  }

  if (action === "join") {
    const project = database.co_creation_projects.find((p) => p.id === body.projectId);
    if (!project) return NextResponse.json({ ok: false, error: "Project not found." }, { status: 404 });
    if (project.status !== "gathering") {
      return NextResponse.json({ ok: false, error: "This project is no longer gathering participants." }, { status: 400 });
    }
    const quantity = Math.min(20, Math.max(1, Number(body.quantity ?? 1)));
    const existing = database.co_creation_members.find((m) => m.project_id === project.id && m.user_id === user.id);
    if (existing) {
      existing.quantity += quantity;
    } else {
      database.co_creation_members.push({
        id: newId("ccm"),
        project_id: project.id,
        user_id: user.id,
        role: "participant",
        quantity,
        payment_status: "reserved",
        joined_at: now(),
      });
      project.supporters += 1;
    }
    project.current_quantity += quantity;
    project.updated_at = now();
    return NextResponse.json({ ok: true, project });
  }

  // create
  if (user.membership_level === "free") {
    return NextResponse.json(
      { ok: false, error: "Starting a co-creation project is a Core Sequence benefit. You can still join existing projects.", upgradeHint: "/membership" },
      { status: 403 }
    );
  }
  const title = String(body.title ?? "").slice(0, 140).trim();
  const concept = String(body.concept ?? "").slice(0, 1200).trim();
  if (title.length < 6 || concept.length < 30) {
    return NextResponse.json({ ok: false, error: "Give the project a title (6+ chars) and a concept (30+ chars)." }, { status: 400 });
  }
  const productTypes = ["wine", "fragrance", "bottle", "giftbox"] as const;
  const project = {
    id: newId("ccp"),
    creator_user_id: user.id,
    title,
    concept,
    product_type: productTypes.includes(body.product_type) ? body.product_type : "wine",
    target_quantity: Math.min(2000, Math.max(10, Number(body.target_quantity ?? 50))),
    current_quantity: 0,
    supporters: 1,
    status: "gathering" as const,
    founder_benefit: "Founder Edition serial + engraving + exclusive QR archive page + founder digital mark",
    public_visible: false,
    review_status: "pending" as const,
    emotion_tags: Array.isArray(body.emotion_tags) ? body.emotion_tags.slice(0, 5).map(String) : [],
    votes: 0,
    created_at: now(),
    updated_at: now(),
  };
  database.co_creation_projects.push(project);
  database.co_creation_members.push({
    id: newId("ccm"),
    project_id: project.id,
    user_id: user.id,
    role: "founder",
    quantity: Math.max(1, Number(body.founder_quantity ?? 1)),
    payment_status: "reserved",
    joined_at: now(),
  });
  database.moderation_logs.push({
    id: newId("mod"),
    user_id: user.id,
    object_id: project.id,
    content_type: "co_creation_project",
    risk_type: "public_display",
    risk_level: "low",
    review_status: "pending",
    reviewer_note: "Auto-queued: new project awaiting platform review before public listing.",
    created_at: now(),
  });

  return NextResponse.json({
    ok: true,
    project,
    note: "Project submitted for platform review. It becomes publicly visible after approval; you can track status in your profile.",
  });
}
