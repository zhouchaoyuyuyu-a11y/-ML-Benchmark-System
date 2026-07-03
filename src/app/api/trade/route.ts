import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { db, newId, now } from "@/lib/store";
import type { TradeRequest } from "@/lib/types";

/** Submit a trade inquiry: quote / authorization / enterprise / collaboration / replenishment. */
export async function POST(req: Request) {
  const user = await getSessionUser();
  const body = await req.json().catch(() => ({}));
  const database = db();

  const types: TradeRequest["request_type"][] = ["quote", "authorization", "enterprise", "collaboration", "replenishment"];
  const requestType = types.includes(body.request_type) ? body.request_type : "quote";

  // Enterprise inquiries may arrive from non-registered contacts via Maison.
  if (!user && requestType !== "enterprise" && requestType !== "collaboration") {
    return NextResponse.json({ ok: false, error: "Sign in to request a quote for your object." }, { status: 401 });
  }
  const contact = String(body.contact ?? user?.email ?? "").slice(0, 200);
  if (!contact) return NextResponse.json({ ok: false, error: "A contact (email or phone) is required." }, { status: 400 });

  const request: TradeRequest = {
    id: newId("trq"),
    user_id: user?.id ?? "usr_guest_lead",
    object_draft_id: body.draftId ? String(body.draftId) : undefined,
    request_type: requestType,
    organization: body.organization ? String(body.organization).slice(0, 200) : undefined,
    contact,
    quantity: Math.max(1, Number(body.quantity ?? 1)),
    budget: String(body.budget ?? "To be discussed").slice(0, 120),
    deadline: body.deadline ? String(body.deadline).slice(0, 40) : undefined,
    delivery_region: body.delivery_region ? String(body.delivery_region).slice(0, 200) : undefined,
    liquid_direction: body.liquid_direction ? String(body.liquid_direction).slice(0, 400) : undefined,
    scent_direction: body.scent_direction ? String(body.scent_direction).slice(0, 400) : undefined,
    bottle_direction: body.bottle_direction ? String(body.bottle_direction).slice(0, 400) : undefined,
    packaging_direction: body.packaging_direction ? String(body.packaging_direction).slice(0, 400) : undefined,
    sample_path: body.sample_path ? String(body.sample_path).slice(0, 300) : undefined,
    invoice_required: Boolean(body.invoice_required),
    logistics_notes: body.logistics_notes ? String(body.logistics_notes).slice(0, 400) : undefined,
    compliance_status: "unchecked",
    human_review_status: "pending",
    quote_status: "none",
    notes: body.notes ? String(body.notes).slice(0, 800) : undefined,
    created_at: now(),
    updated_at: now(),
  };
  database.trade_requests.push(request);

  // Every trade request also opens a concierge lead for follow-up.
  database.concierge_leads.push({
    id: newId("led"),
    user_id: user?.id,
    name: String(body.name ?? user?.nickname ?? "Trade inquiry").slice(0, 120),
    organization: request.organization,
    contact,
    channel: requestType === "enterprise" || requestType === "collaboration" ? "maison" : "trade",
    scenario: [request.request_type, body.scenario].filter(Boolean).join(" · ").slice(0, 300),
    budget: request.budget,
    status: "new",
    created_at: now(),
    updated_at: now(),
  });

  return NextResponse.json({
    ok: true,
    request,
    note: "Received. A human concierge reviews quantity, budget, deadline, region, and compliance before any quotation — expect a reply within one business day.",
  });
}
