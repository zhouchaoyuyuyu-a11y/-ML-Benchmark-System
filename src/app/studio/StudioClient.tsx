"use client";

import Link from "next/link";
import { useState } from "react";
import { Button, ButtonLink, Notice } from "@/components/ui";

type Shape = "tall" | "decanter" | "flask";
type Tone = "ink" | "porcelain" | "blue" | "frosted";
type Accent = "gold" | "silver" | "violet";
type LabelStyle = "vertical" | "band" | "sticker";
type Packaging = "slipcase" | "magnetic" | "kraft";

interface Bi {
  en: string;
  zh: string;
}

interface Geometry {
  bodyW: number;
  bodyH: number;
  neckW: number;
  neckH: number;
  radius: string;
}

interface ToneSpec {
  label: Bi;
  glass: string;
  border: string;
  text: string;
  darkGlass: boolean;
}

interface Mood {
  key: string;
  name: Bi;
  gradient: string;
  top: Bi;
  heart: Bi;
  base: Bi;
}

const SHAPE_LABEL: Record<Shape, Bi> = {
  tall: { en: "Tall cylinder", zh: "高柱瓶" },
  decanter: { en: "Wide decanter", zh: "宽肩醒酒瓶" },
  flask: { en: "Slender flask", zh: "细长瓶" },
};

const GEOMETRY: Record<Shape, Geometry> = {
  tall: { bodyW: 84, bodyH: 200, neckW: 26, neckH: 44, radius: "10px 10px 16px 16px" },
  decanter: { bodyW: 148, bodyH: 148, neckW: 34, neckH: 52, radius: "18px 18px 44px 44px" },
  flask: { bodyW: 62, bodyH: 188, neckW: 20, neckH: 50, radius: "30px 30px 18px 18px" },
};

const TONES: Record<Tone, ToneSpec> = {
  ink: {
    label: { en: "Ink black", zh: "墨黑" },
    glass: "linear-gradient(150deg, #242833 0%, #0b0d13 60%, #05060a 100%)",
    border: "rgba(236,233,226,0.20)",
    text: "#ece9e2",
    darkGlass: true,
  },
  porcelain: {
    label: { en: "Porcelain white", zh: "瓷白" },
    glass: "linear-gradient(150deg, #faf7f0 0%, #e8e3d6 55%, #cfc9ba 100%)",
    border: "rgba(10,12,17,0.28)",
    text: "#211f1a",
    darkGlass: false,
  },
  blue: {
    label: { en: "Ink blue", zh: "墨蓝" },
    glass: "linear-gradient(150deg, #2a3766 0%, #131b3a 55%, #0a0f24 100%)",
    border: "rgba(139,147,255,0.38)",
    text: "#ece9e2",
    darkGlass: true,
  },
  frosted: {
    label: { en: "Frosted", zh: "磨砂" },
    glass: "linear-gradient(150deg, rgba(236,233,226,0.32) 0%, rgba(236,233,226,0.14) 55%, rgba(236,233,226,0.06) 100%)",
    border: "rgba(236,233,226,0.40)",
    text: "#ece9e2",
    darkGlass: true,
  },
};

const ACCENT_LABEL: Record<Accent, Bi> = {
  gold: { en: "Gold", zh: "鎏金" },
  silver: { en: "Silver", zh: "银" },
  violet: { en: "Supply violet", zh: "补给紫" },
};

const ACCENT_HEX: Record<Accent, string> = {
  gold: "#c8a962",
  silver: "#c6ccd6",
  violet: "#8b93ff",
};

const LABEL_STYLE_LABEL: Record<LabelStyle, Bi> = {
  vertical: { en: "Vertical serif", zh: "竖排衬线" },
  band: { en: "Horizontal band", zh: "横向腰封" },
  sticker: { en: "Sticker", zh: "贴纸" },
};

const PACKAGING_LABEL: Record<Packaging, Bi> = {
  slipcase: { en: "Slipcase", zh: "抽拉函套" },
  magnetic: { en: "Magnetic box", zh: "磁吸礼盒" },
  kraft: { en: "Kraft sleeve", zh: "牛皮纸套" },
};

const PACKAGING_DESC: Record<Packaging, Bi> = {
  slipcase: {
    en: "A rigid slipcase with a wax-seal closure. The bottle slides out spine-first, like a book leaving a shelf — restrained, archival, Maison-grade.",
    zh: "硬质函套配火漆封缄，瓶身如取书般抽出——克制、档案感、高定规格。",
  },
  magnetic: {
    en: "A magnetic-clasp box with a night-blue interior and a recessed certificate tray, so the Reserve certificate travels with the object.",
    zh: "磁吸开合礼盒，夜蓝内衬，内嵌证书凹槽——让档案证书随对象同行。",
  },
  kraft: {
    en: "A letterpress-stamped kraft sleeve — lighter, recyclable, with the playful energy of the Supply line.",
    zh: "凸版印章牛皮纸套——更轻、可回收，带着补给线的俏皮气质。",
  },
};

const MOODS: Mood[] = [
  {
    key: "midnight",
    name: { en: "Midnight Order", zh: "午夜秩序" },
    gradient: "linear-gradient(135deg, #0a0c11 0%, #1b2440 55%, #3a4668 100%)",
    top: { en: "Yuzu zest", zh: "柚子皮" },
    heart: { en: "Cold incense", zh: "冷香" },
    base: { en: "Wet stone", zh: "湿石" },
  },
  {
    key: "interval",
    name: { en: "White Interval", zh: "白之间隙" },
    gradient: "linear-gradient(135deg, #f4f1ea 0%, #e4ddcc 55%, #c9b98f 100%)",
    top: { en: "Gardenia", zh: "栀子" },
    heart: { en: "White tea", zh: "白茶" },
    base: { en: "Sandalwood", zh: "檀木" },
  },
  {
    key: "fog",
    name: { en: "City Fog", zh: "城市雾气" },
    gradient: "linear-gradient(135deg, #3c3f52 0%, #565a75 55%, #8b93ff 100%)",
    top: { en: "River mist", zh: "江雾" },
    heart: { en: "Pink pepper", zh: "粉胡椒" },
    base: { en: "Vetiver", zh: "香根草" },
  },
  {
    key: "morale",
    name: { en: "Citrus Morale", zh: "柑橘士气" },
    gradient: "linear-gradient(135deg, #f7b267 0%, #f4845f 55%, #c8553d 100%)",
    top: { en: "Mandarin", zh: "蜜柑" },
    heart: { en: "Sea salt", zh: "海盐" },
    base: { en: "Light musk", zh: "淡麝香" },
  },
  {
    key: "meridian",
    name: { en: "Amber Meridian", zh: "琥珀子午线" },
    gradient: "linear-gradient(135deg, #3a2c18 0%, #7a5a2e 55%, #c8a962 100%)",
    top: { en: "Bergamot", zh: "佛手柑" },
    heart: { en: "Hinoki", zh: "桧木" },
    base: { en: "Amber & warm spice", zh: "琥珀暖香" },
  },
  {
    key: "archive",
    name: { en: "Green Archive", zh: "苍绿档案" },
    gradient: "linear-gradient(135deg, #12241c 0%, #28503c 55%, #5d8a6a 100%)",
    top: { en: "Fig leaf", zh: "无花果叶" },
    heart: { en: "Green tea", zh: "绿茶" },
    base: { en: "Cedar", zh: "雪松" },
  },
];

function Choice({
  label,
  options,
  value,
  onChange,
  zh,
}: {
  label: string;
  options: { value: string; label: Bi }[];
  value: string;
  onChange: (v: string) => void;
  zh: boolean;
}) {
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">{label}</p>
      <div className="mt-2 flex flex-wrap gap-2">
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            onClick={() => onChange(o.value)}
            className={`rounded-md border px-3 py-1.5 text-xs transition-colors ${
              value === o.value
                ? "border-gold/70 bg-gold/10 text-gold"
                : "border-hairline text-mist hover:border-gold/40 hover:text-porcelain"
            }`}
          >
            {zh ? o.label.zh : o.label.en}
          </button>
        ))}
      </div>
    </div>
  );
}

function BottlePlane({
  angle,
  geo,
  tone,
  accent,
  labelStyle,
  copy,
  showLabel,
}: {
  angle: number;
  geo: Geometry;
  tone: ToneSpec;
  accent: string;
  labelStyle: LabelStyle;
  copy: string;
  showLabel: boolean;
}) {
  const bandBg = tone.darkGlass ? "rgba(236,233,226,0.92)" : "rgba(16,18,24,0.88)";
  const bandText = tone.darkGlass ? "#211f1a" : "#ece9e2";

  return (
    <div className="absolute inset-0 flex items-end justify-center pb-8" style={{ transform: `rotateY(${angle}deg)` }}>
      <div className="flex flex-col items-center">
        {/* Cap */}
        <div
          style={{
            width: geo.neckW + 10,
            height: 16,
            borderRadius: 4,
            background: `linear-gradient(160deg, ${accent}, ${accent}55)`,
          }}
        />
        {/* Neck */}
        <div
          style={{
            width: geo.neckW,
            height: geo.neckH,
            background: tone.glass,
            borderLeft: `1px solid ${tone.border}`,
            borderRight: `1px solid ${tone.border}`,
          }}
        />
        {/* Shoulder */}
        <div
          style={{
            width: geo.bodyW,
            height: 20,
            background: tone.glass,
            border: `1px solid ${tone.border}`,
            borderBottom: "none",
            borderRadius: "50% 50% 0 0 / 100% 100% 0 0",
          }}
        />
        {/* Body */}
        <div
          className="relative overflow-hidden"
          style={{
            width: geo.bodyW,
            height: geo.bodyH,
            background: tone.glass,
            border: `1px solid ${tone.border}`,
            borderTop: "none",
            borderRadius: geo.radius,
          }}
        >
          <div
            className="pointer-events-none absolute inset-y-3 left-2 w-1.5 rounded-full"
            style={{ background: "linear-gradient(180deg, rgba(255,255,255,0.35), rgba(255,255,255,0.02))" }}
          />
          <div className="absolute inset-x-3 bottom-2 h-px" style={{ background: accent, opacity: 0.75 }} />

          {showLabel && labelStyle === "vertical" && (
            <div className="absolute inset-0 flex items-center justify-center px-1">
              <p
                className="font-display overflow-hidden"
                style={{
                  writingMode: "vertical-rl",
                  color: tone.text,
                  fontSize: 11,
                  letterSpacing: "0.16em",
                  maxHeight: geo.bodyH - 28,
                  maxWidth: geo.bodyW - 12,
                }}
              >
                {copy}
              </p>
            </div>
          )}

          {showLabel && labelStyle === "band" && (
            <div
              className="absolute inset-x-0 flex items-center justify-center px-1.5"
              style={{
                top: "32%",
                minHeight: 34,
                background: bandBg,
                borderTop: `1.5px solid ${accent}`,
                borderBottom: `1.5px solid ${accent}`,
              }}
            >
              <p className="font-display text-center" style={{ color: bandText, fontSize: 9, lineHeight: 1.3 }}>
                {copy}
              </p>
            </div>
          )}

          {showLabel && labelStyle === "sticker" && (
            <div className="absolute left-1/2 top-1/2" style={{ transform: "translate(-50%, -50%) rotate(-7deg)" }}>
              <div
                style={{
                  background: "rgba(236,233,226,0.95)",
                  border: `2px dashed ${accent}`,
                  borderRadius: 10,
                  padding: "6px 8px",
                  width: geo.bodyW - 10,
                }}
              >
                <p className="text-center" style={{ color: "#211f1a", fontSize: 9, lineHeight: 1.3 }}>
                  {copy}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PackagingMock({ kind, accent }: { kind: Packaging; accent: string }) {
  if (kind === "slipcase") {
    return (
      <div className="relative h-28 w-44 shrink-0">
        <div className="absolute inset-0 rounded-md border bg-obsidian" style={{ borderColor: accent }} />
        <div className="absolute inset-y-2 left-2 w-24 rounded-sm border border-hairline bg-ink" />
        <div className="absolute right-4 top-1/2 h-16 w-5 -translate-y-1/2 rounded-t-full border border-hairline bg-veil" />
        <div className="absolute bottom-2.5 left-5 h-1 w-16 rounded-full" style={{ background: accent, opacity: 0.6 }} />
      </div>
    );
  }
  if (kind === "magnetic") {
    return (
      <div className="flex h-28 w-44 shrink-0 flex-col overflow-hidden rounded-md border bg-obsidian" style={{ borderColor: accent }}>
        <div className="flex h-9 items-center justify-center border-b" style={{ borderColor: accent, background: "rgba(19,27,58,0.6)" }}>
          <span className="h-2 w-2 rounded-full" style={{ background: accent }} />
        </div>
        <div className="flex flex-1 items-center justify-center gap-3">
          <div className="h-12 w-4 rounded-t-full border border-hairline bg-ink" />
          <div className="h-8 w-12 rounded-sm border border-hairline bg-veil" />
        </div>
      </div>
    );
  }
  return (
    <div
      className="flex h-28 w-44 shrink-0 items-center justify-center rounded-md border-2 border-dashed"
      style={{ borderColor: accent, background: "linear-gradient(150deg, #8a6f45, #6f5836)" }}
    >
      <div className="flex h-14 w-14 items-center justify-center rounded-full border-2" style={{ borderColor: "rgba(10,12,17,0.5)" }}>
        <span className="font-display text-xs tracking-[0.2em] text-ink">ZX</span>
      </div>
    </div>
  );
}

interface DraftResponse {
  ok?: boolean;
  error?: string;
  draft?: { id: string; title: string };
}

type SaveState = "idle" | "saving" | "saved" | "gate" | "error";

export default function StudioClient({ zh = false }: { zh?: boolean }) {
  const [shape, setShape] = useState<Shape>("tall");
  const [tone, setTone] = useState<Tone>("ink");
  const [accent, setAccent] = useState<Accent>("gold");
  const [labelStyle, setLabelStyle] = useState<LabelStyle>("vertical");
  const [packaging, setPackaging] = useState<Packaging>("slipcase");
  const [mood, setMood] = useState<Mood>(MOODS[0]);
  const [labelCopy, setLabelCopy] = useState(
    zh ? "世界尽管混乱，今晚我重建自己的秩序。" : "The world can stay chaotic. Tonight, I rebuild my own order."
  );
  const [rotating, setRotating] = useState(true);

  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [saveError, setSaveError] = useState<string | null>(null);
  const [savedTitle, setSavedTitle] = useState<string | null>(null);

  const [cardUrl, setCardUrl] = useState<string | null>(null);
  const [cardLoaded, setCardLoaded] = useState(false);
  const [cardSeq, setCardSeq] = useState(0);

  const t = (b: Bi) => (zh ? b.zh : b.en);
  const geo = GEOMETRY[shape];
  const toneSpec = TONES[tone];
  const accentHex = ACCENT_HEX[accent];

  const trimmedCopy = labelCopy.trim();
  const displayCopy = trimmedCopy === "" ? (zh ? "在这里写一句瓶身文案" : "Write one line of label copy") : trimmedCopy;
  const bottleCopy = displayCopy.length > 64 ? `${displayCopy.slice(0, 64)}…` : displayCopy;

  const visualStyle = [
    SHAPE_LABEL[shape].en,
    `${TONES[tone].label.en} glass`,
    `${ACCENT_LABEL[accent].en} accent`,
    `${LABEL_STYLE_LABEL[labelStyle].en} label`,
    `${PACKAGING_LABEL[packaging].en} packaging`,
    `mood: ${mood.name.en}`,
  ].join(", ");

  async function saveDraft() {
    setSaveState("saving");
    setSaveError(null);
    try {
      const res = await fetch("/api/drafts", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          object_type: "bottle",
          title: `${t(mood.name)} · ${t(SHAPE_LABEL[shape])}`,
          visual_style: visualStyle,
          label_copy: trimmedCopy || undefined,
        }),
      });
      const data = (await res.json().catch(() => ({}))) as DraftResponse;
      if (res.status === 401) {
        setSaveState("gate");
        return;
      }
      if (!res.ok || !data.ok || !data.draft) {
        setSaveError(data.error ?? (zh ? "保存没有成功，请再试一次。" : "That save did not go through — try again."));
        setSaveState("error");
        return;
      }
      setSavedTitle(data.draft.title);
      setSaveState("saved");
    } catch {
      setSaveError(zh ? "网络波动了一下，请再试一次。" : "The connection wobbled — please try again.");
      setSaveState("error");
    }
  }

  function generateCard() {
    const keywords = zh
      ? `${mood.name.zh} · ${mood.top.zh} · ${mood.heart.zh}`
      : `${mood.name.en} · ${mood.top.en} · ${mood.heart.en}`;
    const next = cardSeq + 1;
    setCardSeq(next);
    setCardLoaded(false);
    setCardUrl(
      `/api/card?copy=${encodeURIComponent(displayCopy)}&mark=${encodeURIComponent("Studio Preview")}&keywords=${encodeURIComponent(
        keywords
      )}&seq=${next}`
    );
  }

  return (
    <div className="space-y-6">
      {/* Bottle preview + core controls */}
      <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div className="zx-card p-5 sm:p-6">
          <div className="flex items-center justify-between gap-3">
            <p className="font-display text-lg text-porcelain">{zh ? "铸造预览" : "Casting preview"}</p>
            <label className="flex cursor-pointer items-center gap-2 text-xs text-mist">
              <input
                type="checkbox"
                checked={rotating}
                onChange={(e) => setRotating(e.target.checked)}
                className="h-3.5 w-3.5 accent-gold"
              />
              {zh ? "缓慢旋转" : "Slow rotation"}
            </label>
          </div>

          <div className="relative mx-auto mt-4 flex h-[380px] w-full max-w-xs items-center justify-center" style={{ perspective: "1000px" }}>
            <div className="absolute bottom-8 h-3 w-40 rounded-full bg-gold/15 blur-md" />
            <div
              className={`relative h-[340px] w-[220px] ${rotating ? "zx-rotate-slow" : ""}`}
              style={{ transformStyle: "preserve-3d" }}
            >
              <BottlePlane angle={0} geo={geo} tone={toneSpec} accent={accentHex} labelStyle={labelStyle} copy={bottleCopy} showLabel />
              <BottlePlane angle={90} geo={geo} tone={toneSpec} accent={accentHex} labelStyle={labelStyle} copy={bottleCopy} showLabel={false} />
            </div>
          </div>

          <div className="mt-2 space-y-1 border-t border-hairline pt-3 text-xs text-mist">
            <p>
              {zh ? "瓶型" : "Shape"} · {t(SHAPE_LABEL[shape])} — {zh ? "玻璃" : "Glass"} · {t(TONES[tone].label)}
            </p>
            <p>
              {zh ? "点缀" : "Accent"} · {t(ACCENT_LABEL[accent])} — {zh ? "标签" : "Label"} · {t(LABEL_STYLE_LABEL[labelStyle])}
            </p>
            <p>
              {zh ? "包装" : "Packaging"} · {t(PACKAGING_LABEL[packaging])} — {zh ? "香氛情绪" : "Mood"} · {t(mood.name)}
            </p>
          </div>
        </div>

        <div className="zx-card space-y-5 p-5 sm:p-6">
          <Choice
            label={zh ? "瓶型" : "Bottle shape"}
            options={(Object.keys(SHAPE_LABEL) as Shape[]).map((value) => ({ value, label: SHAPE_LABEL[value] }))}
            value={shape}
            onChange={(v) => setShape(v as Shape)}
            zh={zh}
          />
          <Choice
            label={zh ? "玻璃色调" : "Glass tone"}
            options={(Object.keys(TONES) as Tone[]).map((value) => ({ value, label: TONES[value].label }))}
            value={tone}
            onChange={(v) => setTone(v as Tone)}
            zh={zh}
          />
          <Choice
            label={zh ? "金属点缀" : "Accent"}
            options={(Object.keys(ACCENT_LABEL) as Accent[]).map((value) => ({ value, label: ACCENT_LABEL[value] }))}
            value={accent}
            onChange={(v) => setAccent(v as Accent)}
            zh={zh}
          />
          <Choice
            label={zh ? "标签样式" : "Label style"}
            options={(Object.keys(LABEL_STYLE_LABEL) as LabelStyle[]).map((value) => ({ value, label: LABEL_STYLE_LABEL[value] }))}
            value={labelStyle}
            onChange={(v) => setLabelStyle(v as LabelStyle)}
            zh={zh}
          />
          <div>
            <label htmlFor="studio-copy" className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
              {zh ? "瓶身文案 · 实时渲染" : "Label copy · rendered live"}
            </label>
            <input
              id="studio-copy"
              value={labelCopy}
              onChange={(e) => setLabelCopy(e.target.value)}
              maxLength={80}
              placeholder={zh ? "一句话，会直接出现在瓶身上" : "One sentence — it appears on the bottle as you type"}
              className="mt-2 w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none"
            />
            <p className="mt-1.5 text-xs text-mist">
              {zh
                ? "同一句文案也会用在情绪卡片与档案证书上。"
                : "The same line carries through to the emotional card and the Reserve certificate."}
            </p>
          </div>
        </div>
      </div>

      {/* Packaging + mood board */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="zx-card p-5 sm:p-6">
          <Choice
            label={zh ? "包装预览" : "Packaging preview"}
            options={(Object.keys(PACKAGING_LABEL) as Packaging[]).map((value) => ({ value, label: PACKAGING_LABEL[value] }))}
            value={packaging}
            onChange={(v) => setPackaging(v as Packaging)}
            zh={zh}
          />
          <div className="mt-4 flex flex-col gap-4 sm:flex-row sm:items-center">
            <PackagingMock kind={packaging} accent={accentHex} />
            <div>
              <p className="font-display text-base text-porcelain">{t(PACKAGING_LABEL[packaging])}</p>
              <p className="mt-1.5 text-sm leading-relaxed text-mist">{t(PACKAGING_DESC[packaging])}</p>
            </div>
          </div>
        </div>

        <div className="zx-card p-5 sm:p-6">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
            {zh ? "香氛情绪板 · 六选一" : "Fragrance mood board · choose one of six"}
          </p>
          <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3">
            {MOODS.map((m) => (
              <button
                key={m.key}
                type="button"
                onClick={() => setMood(m)}
                className={`rounded-lg border p-2 text-left transition-colors ${
                  mood.key === m.key ? "border-gold/70" : "border-hairline hover:border-gold/40"
                }`}
              >
                <div className="h-8 w-full rounded-md" style={{ background: m.gradient }} />
                <p className={`mt-1.5 text-xs ${mood.key === m.key ? "text-gold" : "text-mist"}`}>{t(m.name)}</p>
              </button>
            ))}
          </div>
          <div className="mt-4">
            <div className="h-14 w-full rounded-lg" style={{ background: mood.gradient }} />
            <div className="mt-4 flex flex-col items-center gap-1.5">
              <div className="w-32 rounded-sm border border-gold/40 bg-veil px-2 py-1.5 text-center">
                <p className="text-[10px] uppercase tracking-wider text-gold">{zh ? "前调" : "Top"}</p>
                <p className="text-xs text-porcelain">{t(mood.top)}</p>
              </div>
              <div className="w-44 rounded-sm border border-gold/30 bg-veil px-2 py-1.5 text-center">
                <p className="text-[10px] uppercase tracking-wider text-gold">{zh ? "中调" : "Heart"}</p>
                <p className="text-xs text-porcelain">{t(mood.heart)}</p>
              </div>
              <div className="w-56 max-w-full rounded-sm border border-gold/20 bg-veil px-2 py-1.5 text-center">
                <p className="text-[10px] uppercase tracking-wider text-gold">{zh ? "尾调" : "Base"}</p>
                <p className="text-xs text-porcelain">{t(mood.base)}</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="zx-card p-5 sm:p-6">
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={saveDraft} disabled={saveState === "saving"}>
            {saveState === "saving" ? (zh ? "保存中…" : "Saving…") : zh ? "存为草案" : "Save as draft"}
          </Button>
          <Button variant="outline" onClick={generateCard}>
            {zh ? "生成情绪卡片" : "Generate emotional card"}
          </Button>
          <p className="text-xs text-mist">
            {zh
              ? "先保存对象——它是否成为实体，之后在 Trade 里由你决定。"
              : "Save the object first — whether it becomes physical is your call later, in Trade."}
          </p>
        </div>

        {saveState === "error" && saveError && (
          <div className="mt-4">
            <Notice tone="ember">{saveError}</Notice>
          </div>
        )}

        {saveState === "gate" && (
          <div className="mt-4">
            <Notice tone="gold" title={zh ? "差一个免费账号" : "One free account away"}>
              <p>
                {zh
                  ? "草案会存入你的 Design 档案，所以 Studio 需要知道这是谁的档案。注册是免费的。"
                  : "Drafts are saved to your Design archive, so the Studio needs to know whose archive it is. Registering is free."}
              </p>
              <div className="mt-3 flex flex-wrap items-center gap-3">
                <ButtonLink href="/register" variant="gold">
                  {zh ? "免费注册" : "Register free"}
                </ButtonLink>
                <Link href="/login" className="text-xs text-gold hover:underline">
                  {zh ? "已有账号？登录 →" : "Already have one? Sign in →"}
                </Link>
              </div>
            </Notice>
          </div>
        )}

        {saveState === "saved" && savedTitle && (
          <div className="mt-4">
            <Notice tone="gold" title={zh ? "已存入 Design" : "Saved to Design"}>
              <p>
                {zh
                  ? `「${savedTitle}」已保存为草案，并自动生成了带版本哈希的 v1 版本快照。`
                  : `“${savedTitle}” was saved as a draft, and a v1 version snapshot with its own version hash was created automatically.`}
              </p>
              <div className="mt-3 flex flex-wrap gap-3">
                <ButtonLink href="/design" variant="gold">
                  {zh ? "在 Design 中打开" : "Open in Design"}
                </ButtonLink>
                <ButtonLink href="/reserve" variant="outline">
                  {zh ? "了解档案馆" : "About Reserve"}
                </ButtonLink>
              </div>
            </Notice>
          </div>
        )}

        {cardUrl && (
          <div className="mt-5 border-t border-hairline pt-5">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-gold">
              {zh ? "情绪卡片 · 3:4 可分享" : "Emotional card · 3:4, share-ready"}
            </p>
            <div className="mt-3 max-w-xs">
              {!cardLoaded && <div className="zx-skeleton aspect-[3/4] w-full rounded-lg" />}
              <img
                src={cardUrl}
                alt={zh ? "ZOTAIX 情绪卡片预览" : "ZOTAIX emotional card preview"}
                className={`w-full rounded-lg border border-hairline ${cardLoaded ? "" : "hidden"}`}
                onLoad={() => setCardLoaded(true)}
              />
            </div>
            <p className="mt-2 text-xs text-mist">
              {zh
                ? "长按或右键保存图片，即可分享到微信、小红书或任何社交平台。"
                : "Long-press or right-click to save the image, then share it anywhere — WeChat, Instagram, or a group chat."}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
