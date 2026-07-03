// Core AI endpoint: quota check → provider (or Atelier fallback) → usage log
// → conversation persistence. Returns { ok, result: AiResult } or a quota
// message with the upgrade path.

import { cookies } from "next/headers";
import { NextResponse } from "next/server";
import { getSessionUser, VISITOR_COOKIE_NAME } from "@/lib/auth";
import { generateConcept } from "@/lib/ai/provider";
import { checkQuota, consumeQuota } from "@/lib/quota";
import { db, newId, now } from "@/lib/store";
import type { ConciergeInput, ConciergeMode } from "@/lib/types";

const MODES: ConciergeMode[] = ["daily", "gift", "spirit", "fragrance", "copy", "style", "recipient", "co_create", "enterprise"];

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}));
  const mode: ConciergeMode = MODES.includes(body.mode) ? body.mode : "daily";
  const message = String(body.message ?? "").slice(0, 2000);
  if (!message.trim()) {
    return NextResponse.json({ ok: false, error: "Tell the concierge at least one sentence." }, { status: 400 });
  }

  const user = await getSessionUser();
  const cookieStore = await cookies();
  let visitorId = cookieStore.get(VISITOR_COOKIE_NAME)?.value ?? "";
  if (!user && !visitorId) {
    visitorId = newId("vis");
    cookieStore.set(VISITOR_COOKIE_NAME, visitorId, { path: "/", maxAge: 60 * 60 * 24 * 30, sameSite: "lax" });
  }

  const tier = mode === "daily" ? "chat" : mode === "enterprise" ? "creative" : "proposal";
  const decision = checkQuota(user, visitorId, tier);
  if (!decision.allowed) {
    return NextResponse.json(
      { ok: false, quota: true, error: decision.reason, upgradeHint: decision.upgradeHint, tierLabel: decision.tierLabel },
      { status: 429 }
    );
  }

  const input: ConciergeInput = {
    mode,
    message,
    emotion: body.emotion ? String(body.emotion).slice(0, 120) : undefined,
    recipient: body.recipient ? String(body.recipient).slice(0, 120) : undefined,
    scenario: body.scenario ? String(body.scenario).slice(0, 200) : undefined,
    budget: body.budget ? String(body.budget).slice(0, 60) : undefined,
    style: body.style ? String(body.style).slice(0, 120) : undefined,
    locale: body.locale === "zh" ? "zh" : "en",
  };

  const result = await generateConcept(input);
  consumeQuota(user, visitorId, tier, result.model, result.tokens_used);

  // Persist the exchange so it appears in conversation history and admin.
  const database = db();
  const convType = mode === "daily" ? "daily" : mode === "gift" || mode === "recipient" ? "gift" : mode === "fragrance" ? "fragrance" : mode === "spirit" ? "wine" : mode === "enterprise" ? "enterprise" : "product";
  let conversation = body.conversationId
    ? database.conversations.find((c) => c.id === body.conversationId)
    : undefined;
  if (!conversation) {
    conversation = {
      id: newId("cnv"),
      user_id: user?.id,
      visitor_id: user ? undefined : visitorId,
      mode: user ? (user.membership_level === "free" ? "registered" : "member") : "guest",
      conversation_type: convType,
      summary: message.slice(0, 140),
      token_usage: 0,
      created_at: now(),
      updated_at: now(),
    };
    database.conversations.push(conversation);
  }
  conversation.token_usage += result.tokens_used;
  conversation.updated_at = now();
  database.messages.push({
    id: newId("msg"),
    conversation_id: conversation.id,
    role: "user",
    content: message,
    token_usage: Math.round(message.length / 4),
    created_at: now(),
  });
  database.messages.push({
    id: newId("msg"),
    conversation_id: conversation.id,
    role: "assistant",
    content: result.reply,
    structured: result.proposal,
    token_usage: result.tokens_used,
    created_at: now(),
  });

  const after = checkQuota(user, visitorId, tier);
  return NextResponse.json({
    ok: true,
    conversationId: conversation.id,
    result: { ...result, quota_remaining: after.remaining },
  });
}
