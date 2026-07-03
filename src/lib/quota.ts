// Tiered AI quota system. Guests get a small trial; registered users get a
// daily free allowance; members get plan-level allowances. All consumption is
// written to ai_usage_logs so the admin AI-usage module reflects reality.

import { db, newId, now } from "./store";
import type { User } from "./types";

export type AiTier = "chat" | "proposal" | "creative";

export interface QuotaDecision {
  allowed: boolean;
  reason?: string;
  remaining: number | null;
  tierLabel: string;
  upgradeHint?: string;
}

function todayKey(): string {
  return new Date().toISOString().slice(0, 10);
}

function usedToday(userId?: string, visitorId?: string): number {
  const key = todayKey();
  return db()
    .ai_usage_logs.filter((l) => l.created_at.slice(0, 10) === key)
    .filter((l) => (userId ? l.user_id === userId : l.visitor_id === visitorId && !!visitorId))
    .reduce((sum, l) => sum + l.quota_consumed, 0);
}

function monthProposals(userId: string): number {
  const key = new Date().toISOString().slice(0, 7);
  return db()
    .ai_usage_logs.filter((l) => l.user_id === userId && l.created_at.slice(0, 7) === key)
    .filter((l) => l.action_type === "proposal" || l.action_type === "creative")
    .reduce((sum, l) => sum + l.quota_consumed, 0);
}

export function checkQuota(user: User | null, visitorId: string, tier: AiTier): QuotaDecision {
  const s = db().settings;

  if (!user) {
    if (tier !== "chat") {
      return {
        allowed: false,
        remaining: 0,
        tierLabel: "Guest",
        reason: "Structured proposal generation requires a free account.",
        upgradeHint: "/register",
      };
    }
    const used = usedToday(undefined, visitorId);
    const limit = s.guest_daily_chat;
    return {
      allowed: used < limit,
      remaining: Math.max(0, limit - used),
      tierLabel: "Guest trial",
      reason: used >= limit ? "Guest trial limit reached for today — create a free account to continue." : undefined,
      upgradeHint: "/register",
    };
  }

  const level = user.membership_level;
  const dailyLimit =
    level === "pro" ? s.pro_daily_chat : level === "lite" ? s.lite_daily_chat : level === "enterprise" ? 200 : s.free_daily_chat;

  if (tier === "chat") {
    const used = usedToday(user.id);
    return {
      allowed: used < dailyLimit,
      remaining: Math.max(0, dailyLimit - used),
      tierLabel: level === "free" ? "Free" : `Core Sequence ${level[0].toUpperCase()}${level.slice(1)}`,
      reason: used >= dailyLimit ? "Daily Order Energy spent — it renews tomorrow, or leap to a higher sequence." : undefined,
      upgradeHint: "/membership",
    };
  }

  if (tier === "proposal") {
    if (level === "free") {
      const used = usedToday(user.id);
      // Free users may spend daily chat allowance on basic proposals.
      return {
        allowed: used < dailyLimit,
        remaining: Math.max(0, dailyLimit - used),
        tierLabel: "Free",
        reason: used >= dailyLimit ? "Daily allowance reached — Core Sequence unlocks monthly proposal quota." : undefined,
        upgradeHint: "/membership",
      };
    }
    const monthlyLimit = level === "pro" || level === "enterprise" ? s.pro_monthly_proposals : s.lite_monthly_proposals;
    const used = monthProposals(user.id);
    return {
      allowed: used < monthlyLimit,
      remaining: Math.max(0, monthlyLimit - used),
      tierLabel: `Core Sequence ${level[0].toUpperCase()}${level.slice(1)}`,
      reason: used >= monthlyLimit ? "Monthly proposal quota reached — it renews next cycle or via Permission Leap." : undefined,
      upgradeHint: "/membership",
    };
  }

  // creative tier
  if (level === "pro" || level === "enterprise") {
    const used = monthProposals(user.id);
    const monthlyLimit = s.pro_monthly_proposals;
    return {
      allowed: used < monthlyLimit,
      remaining: Math.max(0, monthlyLimit - used),
      tierLabel: "Core Sequence Pro",
      reason: used >= monthlyLimit ? "Monthly creative quota reached." : undefined,
      upgradeHint: "/membership",
    };
  }
  return {
    allowed: false,
    remaining: 0,
    tierLabel: level === "lite" ? "Core Sequence Lite" : "Free",
    reason: "High-cost creative tasks (multi-version proposals, exports, enterprise drafts) require Core Sequence Pro or paid credits.",
    upgradeHint: "/membership",
  };
}

export function consumeQuota(
  user: User | null,
  visitorId: string,
  tier: AiTier,
  model: string,
  tokens: number
): void {
  db().ai_usage_logs.push({
    id: newId("aiu"),
    user_id: user?.id,
    visitor_id: user ? undefined : visitorId || "vis_anonymous",
    model,
    action_type: tier === "chat" ? "chat" : tier === "proposal" ? "proposal" : "creative",
    tokens_used: tokens,
    cost_estimate: Math.round(tokens * 0.00001 * 10000) / 10000,
    quota_consumed: 1,
    created_at: now(),
  });
  if (user) {
    user.used_quota += 1;
    user.updated_at = now();
  }
}
