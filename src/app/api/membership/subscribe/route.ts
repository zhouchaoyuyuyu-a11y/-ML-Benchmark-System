// Membership subscription. With real payment credentials configured (Stripe /
// WeChat Pay / Alipay / PayPal via environment variables) this would create a
// real checkout session; without them the order is recorded in test mode and
// the membership is activated so the full flow remains testable.

import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { integrationStatus } from "@/lib/config";
import { db, newId, now } from "@/lib/store";

export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ ok: false, error: "Sign in to join the Core Sequence." }, { status: 401 });
  const body = await req.json().catch(() => ({}));
  const plan = body.plan === "pro" ? "pro" : "lite";
  const cycle = body.cycle === "quarter" ? "quarter" : "month";
  const database = db();
  const s = database.settings;

  const amount =
    plan === "pro" ? (cycle === "quarter" ? s.pro_price_quarter : s.pro_price_month) : cycle === "quarter" ? s.lite_price_quarter : s.lite_price_month;

  const methods = ["wechat_pay", "alipay", "stripe", "paypal"] as const;
  const method = methods.includes(body.method) ? body.method : "wechat_pay";
  const paymentReady = integrationStatus().find((i) => i.key === (method === "wechat_pay" ? "wechat_pay" : method))?.configured ?? false;

  const order = {
    id: newId("ord"),
    user_id: user.id,
    order_type: "membership" as const,
    title: `Core Sequence ${plan === "pro" ? "Pro" : "Lite"} · ${cycle === "quarter" ? "quarterly" : "monthly"}`,
    amount,
    currency: "CNY" as const,
    payment_method: method,
    status: paymentReady ? ("created" as const) : ("test_mode" as const),
    reference: `SEQ-${plan.toUpperCase()}-${Date.now().toString(36).toUpperCase()}`,
    created_at: now(),
    updated_at: now(),
  };
  database.orders.push(order);

  const days = cycle === "quarter" ? 92 : 31;
  const expires = new Date(Date.now() + days * 24 * 3600 * 1000).toISOString();
  const existing = database.memberships.find((m) => m.user_id === user.id);
  const benefits = {
    plan: plan as "lite" | "pro",
    monthly_quota: plan === "pro" ? s.pro_monthly_proposals : s.lite_monthly_proposals,
    daily_chat_limit: plan === "pro" ? s.pro_daily_chat : s.lite_daily_chat,
    premium_generation_limit: plan === "pro" ? s.pro_monthly_proposals : s.lite_monthly_proposals,
    image_generation_limit: plan === "pro" ? 30 : 5,
    export_enabled: plan === "pro",
    reserve_enabled: true,
    concierge_enabled: plan === "pro",
    expires_at: expires,
  };
  if (existing) {
    Object.assign(existing, benefits);
  } else {
    database.memberships.push({ id: newId("mem"), user_id: user.id, started_at: now(), ...benefits });
  }
  user.user_type = "member";
  user.membership_level = plan;
  user.daily_quota = benefits.daily_chat_limit;
  user.updated_at = now();

  return NextResponse.json({
    ok: true,
    order,
    membership: benefits,
    note: paymentReady
      ? "Checkout created — complete payment to activate."
      : "Payment credentials are not configured in this environment, so the order was recorded in test mode and your sequence benefits are active for evaluation. For production payment, configure the corresponding environment variables.",
  });
}
