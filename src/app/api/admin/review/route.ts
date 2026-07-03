import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { db, newId, now } from "@/lib/store";
import type { ReviewStatus } from "@/lib/types";

const STATUS_MAP: Record<string, ReviewStatus> = {
  approve: "approved",
  reject: "rejected",
  revision: "revision",
  escalate: "escalated",
};

/** Admin review actions across reviewable objects. */
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user || user.user_type !== "admin") {
    return NextResponse.json({ ok: false, error: "Admin access required." }, { status: 403 });
  }
  const body = await req.json().catch(() => ({}));
  const { targetType, targetId } = body;
  const action = String(body.action ?? "");
  const database = db();

  if (targetType === "co_creation_project") {
    const p = database.co_creation_projects.find((x) => x.id === targetId);
    if (!p) return NextResponse.json({ ok: false, error: "Project not found." }, { status: 404 });
    if (action === "hide" || action === "unpublish") {
      p.public_visible = false;
    } else if (action === "feature") {
      p.public_visible = true;
      p.votes += 50;
    } else if (STATUS_MAP[action]) {
      p.review_status = STATUS_MAP[action];
      if (action === "approve") p.public_visible = true;
      if (action === "reject") p.public_visible = false;
    } else {
      return NextResponse.json({ ok: false, error: "Unknown action." }, { status: 400 });
    }
    p.updated_at = now();
  } else if (targetType === "trade_request") {
    const t = database.trade_requests.find((x) => x.id === targetId);
    if (!t) return NextResponse.json({ ok: false, error: "Request not found." }, { status: 404 });
    if (action === "compliance_risk") {
      t.compliance_status = "flagged";
    } else if (action === "infeasible") {
      t.human_review_status = "rejected";
      t.notes = `${t.notes ? t.notes + " · " : ""}Marked supply-chain infeasible by ${user.nickname}`;
    } else if (STATUS_MAP[action]) {
      t.human_review_status = STATUS_MAP[action];
      if (action === "approve") {
        t.compliance_status = "passed";
        t.quote_status = t.quote_status === "none" ? "drafting" : t.quote_status;
      }
    } else {
      return NextResponse.json({ ok: false, error: "Unknown action." }, { status: 400 });
    }
    t.updated_at = now();
  } else if (targetType === "moderation_log") {
    const m = database.moderation_logs.find((x) => x.id === targetId);
    if (!m) return NextResponse.json({ ok: false, error: "Log not found." }, { status: 404 });
    if (!STATUS_MAP[action]) return NextResponse.json({ ok: false, error: "Unknown action." }, { status: 400 });
    m.review_status = STATUS_MAP[action];
    m.reviewer_note = `${m.reviewer_note ? m.reviewer_note + " · " : ""}${action} by ${user.nickname}`;
  } else if (targetType === "object_draft") {
    const d = database.object_drafts.find((x) => x.id === targetId);
    if (!d) return NextResponse.json({ ok: false, error: "Draft not found." }, { status: 404 });
    if (action === "hide" || action === "unpublish") d.public_visible = false;
    else if (action === "feature" || action === "approve") d.public_visible = true;
    else return NextResponse.json({ ok: false, error: "Unknown action." }, { status: 400 });
    d.updated_at = now();
  } else if (targetType === "content_calendar") {
    const c = database.content_calendar.find((x) => x.id === targetId);
    if (!c) return NextResponse.json({ ok: false, error: "Item not found." }, { status: 404 });
    if (action === "approve") c.status = "scheduled";
    else if (action === "feature") c.status = "published";
    else if (action === "reject") c.status = "draft";
    else return NextResponse.json({ ok: false, error: "Unknown action." }, { status: 400 });
    c.updated_at = now();
  } else {
    return NextResponse.json({ ok: false, error: "Unknown target type." }, { status: 400 });
  }

  database.moderation_logs.push({
    id: newId("mod"),
    user_id: user.id,
    object_id: String(targetId),
    content_type: String(targetType),
    risk_type: "public_display",
    risk_level: "low",
    review_status: STATUS_MAP[action] ?? "approved",
    reviewer_note: `Admin action: ${action}`,
    created_at: now(),
  });

  return NextResponse.json({ ok: true });
}
