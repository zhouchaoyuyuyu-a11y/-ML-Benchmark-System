// AI provider adapter. Uses Anthropic or OpenAI when API keys are configured
// (via environment variables only — never hardcoded), and falls back to the
// deterministic Atelier engine otherwise or on any provider failure.

import { aiProvider } from "../config";
import type { AiResult, ConceptProposal, ConciergeInput } from "../types";
import { atelierGenerate } from "./engine";

const SYSTEM_PROMPT = `You are the ZOTAIX concierge: an AI that turns emotions, relationships, scenarios, and budgets into bespoke spirit, fragrance, bottle, and gifting concepts.
Respond ONLY with JSON matching:
{"reply": string, "proposal": {"kind": "daily"|"concept", "emotional_signal": string, "keywords": string[3], "suggestion"?: string, "liquid_direction"?: string, "scent_direction"?: string, "bottle_direction"?: string, "names"?: string[3], "label_copy"?: string, "digital_mark"?: string}}
Rules: keep "reply" under 120 words; label_copy is ONE evocative sentence; never promise production, price, or delivery — these require human confirmation; never target minors; alcohol directions are creative proposals only.`;

function buildUserPrompt(input: ConciergeInput): string {
  return JSON.stringify({
    mode: input.mode,
    message: input.message,
    emotion: input.emotion,
    recipient: input.recipient,
    scenario: input.scenario,
    budget: input.budget,
    style: input.style,
    locale: input.locale ?? "en",
  });
}

function parseProviderJson(raw: string): { reply: string; proposal: Partial<ConceptProposal> } | null {
  try {
    const match = raw.match(/\{[\s\S]*\}/);
    if (!match) return null;
    const parsed = JSON.parse(match[0]);
    if (typeof parsed.reply !== "string" || typeof parsed.proposal !== "object") return null;
    return parsed;
  } catch {
    return null;
  }
}

async function callAnthropic(input: ConciergeInput): Promise<string> {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": process.env.ANTHROPIC_API_KEY ?? "",
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: process.env.ANTHROPIC_MODEL || "claude-sonnet-5",
      max_tokens: 1024,
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: buildUserPrompt(input) }],
    }),
    signal: AbortSignal.timeout(25000),
  });
  if (!res.ok) throw new Error(`anthropic ${res.status}`);
  const data = await res.json();
  return data?.content?.[0]?.text ?? "";
}

async function callOpenAI(input: ConciergeInput): Promise<string> {
  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${process.env.OPENAI_API_KEY ?? ""}`,
    },
    body: JSON.stringify({
      model: process.env.OPENAI_MODEL || "gpt-4o-mini",
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: buildUserPrompt(input) },
      ],
    }),
    signal: AbortSignal.timeout(25000),
  });
  if (!res.ok) throw new Error(`openai ${res.status}`);
  const data = await res.json();
  return data?.choices?.[0]?.message?.content ?? "";
}

export async function generateConcept(input: ConciergeInput): Promise<Omit<AiResult, "quota_remaining">> {
  const provider = aiProvider();
  const fallback = atelierGenerate(input);

  if (provider === "atelier") {
    return {
      reply: fallback.reply,
      proposal: fallback.proposal,
      model: input.mode === "daily" ? "atelier-lite" : "atelier-structured",
      tokens_used: fallback.tokens,
      fallback: true,
    };
  }

  try {
    const raw = provider === "anthropic" ? await callAnthropic(input) : await callOpenAI(input);
    const parsed = parseProviderJson(raw);
    if (!parsed) throw new Error("unparseable provider output");
    // Merge provider output over the Atelier baseline so structured fields
    // are always present even if the provider omits some.
    const proposal: ConceptProposal = {
      ...fallback.proposal,
      ...Object.fromEntries(Object.entries(parsed.proposal).filter(([, v]) => v !== undefined && v !== null && v !== "")),
      next_actions: fallback.proposal.next_actions,
    } as ConceptProposal;
    return {
      reply: parsed.reply,
      proposal,
      model: provider === "anthropic" ? process.env.ANTHROPIC_MODEL || "claude-sonnet-5" : process.env.OPENAI_MODEL || "gpt-4o-mini",
      tokens_used: Math.max(fallback.tokens, Math.round(raw.length / 3)),
      fallback: false,
    };
  } catch {
    return {
      reply: fallback.reply,
      proposal: fallback.proposal,
      model: "atelier-structured (provider fallback)",
      tokens_used: fallback.tokens,
      fallback: true,
    };
  }
}
