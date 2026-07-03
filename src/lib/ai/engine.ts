// ZOTAIX Atelier — deterministic structured concept engine.
// This is the graceful fallback when no external AI provider is configured,
// and the guaranteed source of structured fields when a provider returns
// malformed output. It maps emotion/recipient/scenario/budget signals onto
// curated creative banks so every generation is coherent and on-brand.

import type { ConceptProposal, ConciergeInput, ConciergeMode } from "../types";

interface Palette {
  keys: string[];
  signal: string;
  liquid: string;
  scent: string;
  bottle: string;
  names: string[];
  copy: string;
  mark: string;
  suggestion: string;
}

const PALETTES: Palette[] = [
  {
    keys: ["tired", "exhaust", "overwork", "burnout", "stress", "pressure", "累", "疲惫", "加班", "压力"],
    signal: "Depleted but unbroken — the engine wants quiet fuel, not fireworks",
    liquid: "Low-ABV oolong-infused base with smoked plum and a soft honey finish (around 12–18% ABV)",
    scent: "Cold incense and cedar over warm rice steam, a trace of yuzu at the top",
    bottle: "Matte ink-black cylinder, champagne-gold vertical typesetting, weighted base",
    names: ["Quiet Foundry", "Order 03:00", "Slow Voltage"],
    copy: "The world can stay chaotic. Tonight, I rebuild my own order.",
    mark: "Night Rebuilder Mark",
    suggestion: "Protect twenty unscheduled minutes tonight. Small rituals are load-bearing.",
  },
  {
    keys: ["breakup", "heartbreak", "lost", "miss", "lonely", "分手", "失恋", "想念", "孤独"],
    signal: "A closing chapter asking to be archived with dignity, not deleted",
    liquid: "Clear plum spirit with bitter-orange peel — clean start, lingering warmth (20% ABV)",
    scent: "Rain-washed white tea, salt air, the last note a dry woody ember",
    bottle: "Frosted glass with a fading gradient wrap, single seam of silver foil",
    names: ["Low Tide Archive", "The Unsent Draft", "After Rain, Ledger"],
    copy: "Filed, not forgotten. The next page is already lighter.",
    mark: "Tide-Turn Mark",
    suggestion: "Write one sentence to the past version of you, then seal it into a card.",
  },
  {
    keys: ["exam", "study", "test", "thesis", "deadline", "考试", "考研", "论文", "复习"],
    signal: "Compressed focus — morale supply matters more than celebration",
    liquid: "Zero-proof sparkling ration: citrus, sea salt, cold-brew jasmine",
    scent: "Sharp bergamot over graphite and clean paper",
    bottle: "Sticker-energy label on a stout can-style bottle, sunset-orange gradient",
    names: ["Night Study Ration", "Page Turner", "Last Sprint Soda"],
    copy: "One more page. Then the whole sky.",
    mark: "Persistence Mark · Exam Season",
    suggestion: "Batch your worry into a 10-minute window, then trade it for one section done.",
  },
  {
    keys: ["birthday", "celebrate", "party", "anniversary", "wedding", "生日", "庆祝", "纪念日", "婚礼"],
    signal: "A dated moment that deserves a serial number and a toast",
    liquid: "Sparkling plum-blossom blend with a golden pear heart (8% ABV, crowd-friendly)",
    scent: "Champagne accord, white peach, a candle-smoke whisper at the end",
    bottle: "Ivory wrap with foil-stamped date medallion, ribbon-ready neck",
    names: ["Annual Light", "The Dated Bottle", "Candle Arithmetic"],
    copy: "Some days get numbers. This one gets a name.",
    mark: "Celebration Seal",
    suggestion: "Put the date on the label — dated objects become anniversaries automatically.",
  },
  {
    keys: ["thank", "gratitude", "mentor", "client", "boss", "appreciation", "感谢", "恩师", "客户", "答谢"],
    signal: "Gratitude that wants weight — formal, warm, precisely worded",
    liquid: "Aged huangjiu blend with citrus-peel finish, decanted small-batch",
    scent: "Amber, hinoki, and a library's worth of warm paper",
    bottle: "Wide-shoulder decanter, bronze cap, embossed monogram wrap",
    names: ["Meridian Thanks", "The Long Ledger", "Bearing & Gratitude"],
    copy: "A year measured in arrivals. Thank you for returning.",
    mark: "Gratitude Seal · Bronze",
    suggestion: "Name the specific moment you're thankful for — precision is the luxury.",
  },
  {
    keys: ["love", "partner", "crush", "romance", "date", "恋人", "喜欢", "爱", "告白"],
    signal: "Steady light rather than fireworks — devotion in a quiet register",
    liquid: "White-tea infusion with gardenia honey, barely sweet (10% ABV)",
    scent: "Gardenia and white tea over sandalwood",
    bottle: "Matte porcelain white, tall slender profile, one engraved line at the base",
    names: ["White Interval", "Eleventh Light", "Quiet Bloom"],
    copy: "Some light does not flare. It stays.",
    mark: "Devotion Mark · Porcelain",
    suggestion: "One engraved line beats a paragraph. Choose the sentence you'd say twice.",
  },
  {
    keys: ["friend", "brother", "sister", "roommate", "team", "朋友", "兄弟", "闺蜜", "同事"],
    signal: "Companionship with inside jokes — playful outside, loyal inside",
    liquid: "Session-strength citrus shandy with a green-tea snap (4% ABV)",
    scent: "Grapefruit zest, cut grass, a mineral cool-down",
    bottle: "Two-tone label split down the middle — one half for each of you",
    names: ["Split Label", "The Standing Appointment", "Same Table Energy"],
    copy: "No occasion. That's the occasion.",
    mark: "Same-Table Mark",
    suggestion: "Make it a pair — matching serials read like a handshake.",
  },
  {
    keys: ["city", "travel", "hometown", "souvenir", "chongqing", "kyoto", "城市", "旅行", "故乡", "重庆"],
    signal: "A place carried as weather — fog, neon, river air",
    liquid: "Pepper-warm baijiu base softened with pomelo and river-cool mint",
    scent: "River fog, Sichuan pepper warmth, neon-wet asphalt sweetness",
    bottle: "Topographic-line label in gunmetal and coral, skyline die-cut",
    names: ["City Fog", "River Delta Diary", "Neon Altitude"],
    copy: "Every city is a weather you learn to miss.",
    mark: "Cartographer's Mark",
    suggestion: "Anchor it to one street, not the whole city — memory lives at street level.",
  },
];

const DEFAULT_PALETTE: Palette = {
  keys: [],
  signal: "An open signal — curiosity with room to be shaped",
  liquid: "Balanced botanical base: elderflower, green plum, faint juniper (15% ABV)",
  scent: "Bergamot opening into warm woods and clean musk",
  bottle: "Ink-blue glass with a champagne-gold meridian ring, minimal serif label",
  names: ["First Draft", "Open Meridian", "Prelude 001"],
  copy: "Begin anywhere. The order will follow.",
  mark: "Prelude Mark",
  suggestion: "Give the concierge one feeling and one person — everything else can be generated.",
};

function pickPalette(text: string): Palette {
  const lower = text.toLowerCase();
  let best: Palette | null = null;
  let bestScore = 0;
  for (const p of PALETTES) {
    const score = p.keys.reduce((acc, k) => acc + (lower.includes(k) ? 1 : 0), 0);
    if (score > bestScore) {
      best = p;
      bestScore = score;
    }
  }
  return best ?? DEFAULT_PALETTE;
}

function keywordsFrom(input: ConciergeInput, palette: Palette): string[] {
  const kws: string[] = [];
  if (palette !== DEFAULT_PALETTE) kws.push(palette.names[0].toLowerCase());
  if (input.emotion) kws.push(input.emotion.toLowerCase());
  if (input.scenario) kws.push(input.scenario.toLowerCase());
  if (input.style) kws.push(input.style.toLowerCase());
  if (kws.length === 0) kws.push("open signal", "first draft");
  return [...new Set(kws)].slice(0, 3);
}

const MODE_ACTIONS: Record<ConciergeMode, ConceptProposal["next_actions"]> = {
  daily: ["save_inspiration", "emotional_card", "add_preferences", "share"],
  gift: ["save_inspiration", "gift_draft", "label_copy", "human_concierge"],
  spirit: ["save_inspiration", "gift_draft", "co_create", "physical_casting"],
  fragrance: ["save_inspiration", "gift_draft", "co_create", "physical_casting"],
  copy: ["save_inspiration", "label_copy", "emotional_card", "share"],
  style: ["save_inspiration", "add_preferences", "emotional_card", "share"],
  recipient: ["save_inspiration", "gift_draft", "add_preferences", "human_concierge"],
  co_create: ["save_inspiration", "co_create", "share", "human_concierge"],
  enterprise: ["save_inspiration", "gift_draft", "human_concierge", "physical_casting"],
};

export function atelierGenerate(input: ConciergeInput): { reply: string; proposal: ConceptProposal; tokens: number } {
  const context = [input.message, input.emotion, input.recipient, input.scenario, input.style].filter(Boolean).join(" · ");
  const palette = pickPalette(context);
  const keywords = keywordsFrom(input, palette);
  const actions = MODE_ACTIONS[input.mode] ?? MODE_ACTIONS.daily;

  if (input.mode === "daily") {
    const reply =
      input.locale === "zh"
        ? `收到你的信号：${palette.signal}。今天不需要宏大计划——${palette.suggestion} 如果想把这份状态留下来，可以生成一张情绪卡片。`
        : `Signal received: ${palette.signal}. No grand plan needed today — ${palette.suggestion} If this state is worth keeping, turn it into an emotional card.`;
    return {
      reply,
      proposal: {
        kind: "daily",
        emotional_signal: palette.signal,
        keywords,
        suggestion: palette.suggestion,
        label_copy: palette.copy,
        digital_mark: palette.mark,
        next_actions: actions,
      },
      tokens: 220,
    };
  }

  const recipientLine = input.recipient ? ` for ${input.recipient}` : "";
  const budgetLine = input.budget ? ` within ${input.budget}` : "";
  const reply =
    input.locale === "zh"
      ? `基于你的描述，我起了一版完整的定制方向${input.recipient ? `（对象：${input.recipient}）` : ""}：酒体、香氛、瓶身与文案都在下方的结构卡片里。所有内容均为创意提案，实体化需人工确认与最终报价。`
      : `From your brief, here is a full customization direction${recipientLine}${budgetLine}: liquid, fragrance, bottle, and copy are in the structured card below. Everything is a creative proposal — physical delivery requires human confirmation and a final quotation.`;

  return {
    reply,
    proposal: {
      kind: "concept",
      emotional_signal: palette.signal,
      keywords,
      liquid_direction: palette.liquid,
      scent_direction: palette.scent,
      bottle_direction: palette.bottle,
      names: palette.names,
      label_copy: palette.copy,
      digital_mark: palette.mark,
      next_actions: actions,
    },
    tokens: 420,
  };
}
