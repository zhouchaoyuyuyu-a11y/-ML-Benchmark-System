"use client";

import Link from "next/link";
import { useState, type FormEvent, type ReactNode } from "react";
import { Button, Card, Notice, StatusPill, Tag } from "@/components/ui";
import { profileNotice } from "@/lib/copy";
import type {
  Conversation,
  ObjectDraft,
  RelationshipProfile,
  ReserveRecord,
  UserProfile,
} from "@/lib/types";

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

const MBTI_TYPES = [
  "INTJ", "INTP", "ENTJ", "ENTP",
  "INFJ", "INFP", "ENFJ", "ENFP",
  "ISTJ", "ISFJ", "ESTJ", "ESFJ",
  "ISTP", "ISFP", "ESTP", "ESFP",
];

const ZODIACS: { value: string; zh: string }[] = [
  { value: "Aries", zh: "白羊座" },
  { value: "Taurus", zh: "金牛座" },
  { value: "Gemini", zh: "双子座" },
  { value: "Cancer", zh: "巨蟹座" },
  { value: "Leo", zh: "狮子座" },
  { value: "Virgo", zh: "处女座" },
  { value: "Libra", zh: "天秤座" },
  { value: "Scorpio", zh: "天蝎座" },
  { value: "Sagittarius", zh: "射手座" },
  { value: "Capricorn", zh: "摩羯座" },
  { value: "Aquarius", zh: "水瓶座" },
  { value: "Pisces", zh: "双鱼座" },
];

const BLOOD_TYPES = ["A", "B", "AB", "O"];
const AGE_RANGES = ["18-24", "25-34", "35-44", "45-54", "55+"];

const TOLERANCES: { value: string; en: string; zh: string }[] = [
  { value: "None", en: "None — zero-proof only", zh: "不饮酒——只要零酒精" },
  { value: "Light", en: "Light", zh: "浅尝" },
  { value: "Moderate", en: "Moderate", zh: "适中" },
  { value: "High", en: "High", zh: "海量" },
];

const VISUAL_STYLES: { value: string; en: string; zh: string }[] = [
  { value: "Eastern", en: "Eastern", zh: "东方" },
  { value: "Futuristic", en: "Futuristic", zh: "未来" },
  { value: "Retro", en: "Retro", zh: "复古" },
  { value: "Minimal", en: "Minimal", zh: "极简" },
  { value: "Sweet-cool", en: "Sweet-cool", zh: "甜酷" },
  { value: "Luxury", en: "Luxury", zh: "奢华" },
  { value: "Restrained", en: "Restrained", zh: "克制" },
  { value: "Playful", en: "Playful", zh: "俏皮" },
];

const RELATION_TYPES: { value: string; en: string; zh: string }[] = [
  { value: "Partner", en: "Partner", zh: "伴侣" },
  { value: "Close friend", en: "Close friend", zh: "挚友" },
  { value: "Family", en: "Family", zh: "家人" },
  { value: "Colleague", en: "Colleague", zh: "同事" },
  { value: "Mentor", en: "Mentor", zh: "导师" },
  { value: "Client", en: "Client", zh: "客户" },
  { value: "Other", en: "Other", zh: "其他" },
];

interface ProfileForm {
  mbti: string;
  zodiac: string;
  blood_type: string;
  age_range: string;
  nickname: string;
  address_style: string;
  favorite_colors: string;
  scent_preferences: string;
  alcohol_preferences: string;
  alcohol_tolerance: string;
  non_alcohol_ok: boolean;
  music: string;
  movies: string;
  cities: string;
  literary_imagery: string;
  visual_preferences: string[];
  gift_preferences: string;
  budget_range: string;
  emotional_state: string;
  common_scenarios: string;
  memory_enabled: boolean;
  privacy_level: "private" | "co-create" | "public";
}

function joinList(v?: string[]): string {
  return (v ?? []).join(", ");
}

function toList(s: string): string[] {
  return s
    .split(/[,，、]/)
    .map((x) => x.trim())
    .filter(Boolean)
    .slice(0, 12);
}

function fromProfile(p: UserProfile | null): ProfileForm {
  return {
    mbti: p?.mbti ?? "",
    zodiac: p?.zodiac ?? "",
    blood_type: p?.blood_type ?? "",
    age_range: p?.age_range ?? "",
    nickname: p?.nickname ?? "",
    address_style: p?.address_style ?? "",
    favorite_colors: joinList(p?.favorite_colors),
    scent_preferences: joinList(p?.scent_preferences),
    alcohol_preferences: joinList(p?.alcohol_preferences),
    alcohol_tolerance: p?.alcohol_tolerance ?? "",
    non_alcohol_ok: p?.non_alcohol_ok ?? true,
    music: p?.music ?? "",
    movies: p?.movies ?? "",
    cities: p?.cities ?? "",
    literary_imagery: p?.literary_imagery ?? "",
    visual_preferences: p?.visual_preferences ?? [],
    gift_preferences: joinList(p?.gift_preferences),
    budget_range: p?.budget_range ?? "",
    emotional_state: p?.emotional_state ?? "",
    common_scenarios: joinList(p?.common_scenarios),
    memory_enabled: p?.memory_enabled ?? true,
    privacy_level: p?.privacy_level ?? "private",
  };
}

interface RelForm {
  relation_type: string;
  nickname: string;
  age_range: string;
  preferences: string;
  important_dates: string;
  notes: string;
}

const emptyRel: RelForm = {
  relation_type: "Close friend",
  nickname: "",
  age_range: "",
  preferences: "",
  important_dates: "",
  notes: "",
};

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs uppercase tracking-wider text-mist">{label}</span>
      <span className="mt-1.5 block">{children}</span>
    </label>
  );
}

function SectionCard({
  step,
  title,
  description,
  children,
}: {
  step: string;
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <Card>
      <div className="flex items-start gap-3">
        <span className="font-display text-2xl text-gold/60">{step}</span>
        <div>
          <h2 className="font-display text-lg text-porcelain">{title}</h2>
          {description && <p className="mt-1 text-sm leading-relaxed text-mist">{description}</p>}
        </div>
      </div>
      <div className="mt-5">{children}</div>
    </Card>
  );
}

export default function ProfileClient({
  zh = false,
  profile,
  relationships,
  drafts,
  reserve,
  conversations,
}: {
  zh?: boolean;
  profile: UserProfile | null;
  relationships: RelationshipProfile[];
  drafts: ObjectDraft[];
  reserve: ReserveRecord[];
  conversations: Conversation[];
}) {
  const [form, setForm] = useState<ProfileForm>(() => fromProfile(profile));
  const [rels, setRels] = useState<RelationshipProfile[]>(relationships);
  const [relForm, setRelForm] = useState<RelForm>(emptyRel);
  const [saving, setSaving] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleted, setDeleted] = useState(false);

  function set<K extends keyof ProfileForm>(key: K, value: ProfileForm[K]) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  function toggleVisual(value: string) {
    setForm((f) => ({
      ...f,
      visual_preferences: f.visual_preferences.includes(value)
        ? f.visual_preferences.filter((v) => v !== value)
        : [...f.visual_preferences, value],
    }));
  }

  async function saveSection(section: string, payload: Record<string, unknown>) {
    setSaving(section);
    setError(null);
    try {
      const res = await fetch("/api/profile", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = (await res.json().catch(() => ({}))) as { ok?: boolean; error?: string };
      if (!res.ok || !data.ok) {
        setError(data.error ?? (zh ? "保存没有成功，请再试一次。" : "That save did not go through — please try again."));
        return;
      }
      setDeleted(false);
      setSaved(section);
      window.setTimeout(() => setSaved((s) => (s === section ? null : s)), 3000);
    } catch {
      setError(zh ? "网络波动了一下，请再试一次。" : "The connection wobbled — please try again.");
    } finally {
      setSaving(null);
    }
  }

  function saveButton(section: string, onClick: () => void) {
    return (
      <div className="mt-5 flex items-center gap-3">
        <Button onClick={onClick} disabled={saving === section}>
          {saving === section ? (zh ? "保存中…" : "Saving…") : zh ? "保存" : "Save"}
        </Button>
        {saved === section && (
          <span className="text-sm text-jade" role="status">
            ✓ {zh ? "已保存" : "Saved"}
          </span>
        )}
      </div>
    );
  }

  function saveIdentity() {
    void saveSection("identity", {
      mbti: form.mbti,
      zodiac: form.zodiac,
      blood_type: form.blood_type,
      age_range: form.age_range,
      nickname: form.nickname,
      address_style: form.address_style,
    });
  }

  function savePreferences() {
    void saveSection("preferences", {
      favorite_colors: toList(form.favorite_colors),
      scent_preferences: toList(form.scent_preferences),
      alcohol_preferences: toList(form.alcohol_preferences),
      alcohol_tolerance: form.alcohol_tolerance,
      non_alcohol_ok: form.non_alcohol_ok,
      music: form.music,
      movies: form.movies,
      cities: form.cities,
      literary_imagery: form.literary_imagery,
      visual_preferences: form.visual_preferences,
      gift_preferences: toList(form.gift_preferences),
      budget_range: form.budget_range,
      emotional_state: form.emotional_state,
      common_scenarios: toList(form.common_scenarios),
    });
  }

  function savePrivacy() {
    void saveSection("privacy", {
      memory_enabled: form.memory_enabled,
      privacy_level: form.privacy_level,
    });
  }

  async function addRelationship(e: FormEvent) {
    e.preventDefault();
    if (!relForm.nickname.trim()) {
      setError(zh ? "先给这位重要的人一个称呼。" : "Give this person a nickname first.");
      return;
    }
    setSaving("relationship");
    setError(null);
    try {
      const res = await fetch("/api/profile", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ relationship: relForm }),
      });
      const data = (await res.json().catch(() => ({}))) as { ok?: boolean; error?: string };
      if (!res.ok || !data.ok) {
        setError(data.error ?? (zh ? "添加没有成功，请再试一次。" : "That did not go through — please try again."));
        return;
      }
      const refreshed = await fetch("/api/profile");
      const refreshedData = (await refreshed.json().catch(() => ({}))) as {
        ok?: boolean;
        relationships?: RelationshipProfile[];
      };
      if (refreshedData.ok && Array.isArray(refreshedData.relationships)) {
        setRels(refreshedData.relationships);
      }
      setRelForm(emptyRel);
      setSaved("relationship");
      window.setTimeout(() => setSaved((s) => (s === "relationship" ? null : s)), 3000);
    } catch {
      setError(zh ? "网络波动了一下，请再试一次。" : "The connection wobbled — please try again.");
    } finally {
      setSaving(null);
    }
  }

  async function deleteProfileData() {
    setSaving("delete");
    setError(null);
    try {
      const res = await fetch("/api/profile", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action: "delete" }),
      });
      const data = (await res.json().catch(() => ({}))) as { ok?: boolean; error?: string };
      if (!res.ok || !data.ok) {
        setError(data.error ?? (zh ? "删除没有成功，请再试一次。" : "That deletion did not go through — please try again."));
        return;
      }
      setForm(fromProfile(null));
      setDeleted(true);
      setConfirmDelete(false);
    } catch {
      setError(zh ? "网络波动了一下，请再试一次。" : "The connection wobbled — please try again.");
    } finally {
      setSaving(null);
    }
  }

  function exportArchive() {
    const payload = {
      exported_at: new Date().toISOString(),
      order_hub_profile: {
        mbti: form.mbti || null,
        zodiac: form.zodiac || null,
        blood_type: form.blood_type || null,
        age_range: form.age_range || null,
        nickname: form.nickname || null,
        address_style: form.address_style || null,
        favorite_colors: toList(form.favorite_colors),
        scent_preferences: toList(form.scent_preferences),
        alcohol_preferences: toList(form.alcohol_preferences),
        alcohol_tolerance: form.alcohol_tolerance || null,
        non_alcohol_ok: form.non_alcohol_ok,
        music: form.music || null,
        movies: form.movies || null,
        cities: form.cities || null,
        literary_imagery: form.literary_imagery || null,
        visual_preferences: form.visual_preferences,
        gift_preferences: toList(form.gift_preferences),
        budget_range: form.budget_range || null,
        emotional_state: form.emotional_state || null,
        common_scenarios: toList(form.common_scenarios),
        memory_enabled: form.memory_enabled,
        privacy_level: form.privacy_level,
      },
      relationship_profiles: rels,
      inspiration_drafts: drafts,
      reserve_records: reserve,
      generation_history: conversations,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `zotaix-archive-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    setSaved("export");
    window.setTimeout(() => setSaved((s) => (s === "export" ? null : s)), 3000);
  }

  const notSet = zh ? "未设置" : "Not set";

  return (
    <div className="space-y-6">
      {error && <Notice tone="ember">{error}</Notice>}
      {deleted && (
        <Notice tone="gold" title={zh ? "档案数据已删除" : "Profile data deleted"}>
          {zh
            ? "你的表达标签与偏好已从中枢移除。下方表单已清空——任何时候都可以重新填写。"
            : "Your expression tags and preferences have been removed from the hub. The forms below are cleared — refill them whenever you like."}
        </Notice>
      )}

      {/* 1 · Self-expression tags */}
      <SectionCard
        step="01"
        title={zh ? "自我表达标签" : "Self-expression tags"}
        description={
          zh
            ? "这些标签只影响礼宾说话的语气与意象，不影响权限或价格。"
            : "These tags shape the concierge's tone and imagery — never access or price."
        }
      >
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <Tag tone="gold">{zh ? "仅作表达，不作诊断" : "Self-expression only, not diagnosis"}</Tag>
        </div>
        <Notice tone="gold">{zh ? profileNotice.zh : profileNotice.en}</Notice>
        <div className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Field label="MBTI">
            <select value={form.mbti} onChange={(e) => set("mbti", e.target.value)} className={inputCls}>
              <option value="">{notSet}</option>
              {MBTI_TYPES.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </Field>
          <Field label={zh ? "星座" : "Zodiac"}>
            <select value={form.zodiac} onChange={(e) => set("zodiac", e.target.value)} className={inputCls}>
              <option value="">{notSet}</option>
              {ZODIACS.map((z) => (
                <option key={z.value} value={z.value}>
                  {zh ? z.zh : z.value}
                </option>
              ))}
            </select>
          </Field>
          <Field label={zh ? "血型" : "Blood type"}>
            <select value={form.blood_type} onChange={(e) => set("blood_type", e.target.value)} className={inputCls}>
              <option value="">{notSet}</option>
              {BLOOD_TYPES.map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </select>
          </Field>
          <Field label={zh ? "年龄段" : "Age range"}>
            <select value={form.age_range} onChange={(e) => set("age_range", e.target.value)} className={inputCls}>
              <option value="">{notSet}</option>
              {AGE_RANGES.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </Field>
          <Field label={zh ? "昵称" : "Nickname"}>
            <input
              value={form.nickname}
              onChange={(e) => set("nickname", e.target.value)}
              placeholder={zh ? "礼宾如何称呼你" : "What the concierge calls you"}
              className={inputCls}
              maxLength={60}
            />
          </Field>
          <Field label={zh ? "称呼方式" : "Address style"}>
            <input
              value={form.address_style}
              onChange={(e) => set("address_style", e.target.value)}
              placeholder={zh ? "例如：直呼其名，不要敬语" : "e.g. First name, no honorifics"}
              className={inputCls}
              maxLength={120}
            />
          </Field>
        </div>
        {saveButton("identity", saveIdentity)}
      </SectionCard>

      {/* 2 · Preferences */}
      <SectionCard
        step="02"
        title={zh ? "偏好档案" : "Preferences"}
        description={
          zh
            ? "列表字段用逗号分隔多个条目。写得越具体，提案越像你。"
            : "Separate multiple entries with commas. The more specific you are, the more the proposals sound like you."
        }
      >
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label={zh ? "喜欢的颜色" : "Favorite colors"}>
            <input
              value={form.favorite_colors}
              onChange={(e) => set("favorite_colors", e.target.value)}
              placeholder={zh ? "墨蓝, 香槟金" : "Ink blue, champagne gold"}
              className={inputCls}
            />
          </Field>
          <Field label={zh ? "香气偏好" : "Scent preferences"}>
            <input
              value={form.scent_preferences}
              onChange={(e) => set("scent_preferences", e.target.value)}
              placeholder={zh ? "木质, 冷香, 白茶" : "Woody, cold incense, white tea"}
              className={inputCls}
            />
          </Field>
          <Field label={zh ? "酒饮偏好" : "Alcohol preferences"}>
            <input
              value={form.alcohol_preferences}
              onChange={(e) => set("alcohol_preferences", e.target.value)}
              placeholder={zh ? "单一麦芽, 梅酒, 低度植物酒" : "Single malt, umeshu, low-ABV botanical"}
              className={inputCls}
            />
          </Field>
          <Field label={zh ? "酒量 / 耐受度" : "Alcohol tolerance"}>
            <select
              value={form.alcohol_tolerance}
              onChange={(e) => set("alcohol_tolerance", e.target.value)}
              className={inputCls}
            >
              <option value="">{notSet}</option>
              {TOLERANCES.map((t) => (
                <option key={t.value} value={t.value}>
                  {zh ? t.zh : t.en}
                </option>
              ))}
            </select>
          </Field>
        </div>
        <div className="mt-4 rounded-lg border border-hairline bg-obsidian/60 px-4 py-3">
          <label className="flex cursor-pointer items-start gap-3">
            <input
              type="checkbox"
              checked={form.non_alcohol_ok}
              onChange={(e) => set("non_alcohol_ok", e.target.checked)}
              className="mt-0.5 h-4 w-4 accent-gold"
            />
            <span>
              <span className="text-sm font-medium text-porcelain">
                {zh ? "接受零酒精方案" : "Zero-proof proposals welcome"}
              </span>
              <span className="mt-0.5 block text-xs leading-relaxed text-mist">
                {zh
                  ? "开启后，礼宾会在合适的场景优先给出零酒精 / 低酒精方向。"
                  : "When on, the concierge favors zero- and low-proof directions where the scenario allows."}
              </span>
            </span>
          </label>
        </div>
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <Field label={zh ? "音乐" : "Music"}>
            <input
              value={form.music}
              onChange={(e) => set("music", e.target.value)}
              placeholder={zh ? "后摇, 坂本龙一" : "Post-rock, Ryuichi Sakamoto"}
              className={inputCls}
            />
          </Field>
          <Field label={zh ? "电影" : "Movies"}>
            <input
              value={form.movies}
              onChange={(e) => set("movies", e.target.value)}
              placeholder={zh ? "花样年华, 降临" : "In the Mood for Love, Arrival"}
              className={inputCls}
            />
          </Field>
          <Field label={zh ? "城市" : "Cities"}>
            <input
              value={form.cities}
              onChange={(e) => set("cities", e.target.value)}
              placeholder={zh ? "京都, 重庆, 雷克雅未克" : "Kyoto, Chongqing, Reykjavik"}
              className={inputCls}
            />
          </Field>
          <Field label={zh ? "文学意象" : "Literary imagery"}>
            <input
              value={form.literary_imagery}
              onChange={(e) => set("literary_imagery", e.target.value)}
              placeholder={zh ? "博尔赫斯的图书馆, 午夜的港口" : "Borges' library, midnight harbors"}
              className={inputCls}
            />
          </Field>
        </div>
        <div className="mt-4">
          <p className="text-xs uppercase tracking-wider text-mist">{zh ? "视觉风格（可多选）" : "Visual style (pick any)"}</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {VISUAL_STYLES.map((v) => {
              const active = form.visual_preferences.includes(v.value);
              return (
                <button
                  key={v.value}
                  type="button"
                  onClick={() => toggleVisual(v.value)}
                  aria-pressed={active}
                  className={`rounded-full border px-3 py-1.5 text-xs transition-colors ${
                    active ? "border-gold bg-gold/10 text-gold" : "border-hairline text-mist hover:text-porcelain"
                  }`}
                >
                  {zh ? v.zh : v.en}
                </button>
              );
            })}
          </div>
        </div>
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <Field label={zh ? "礼物偏好" : "Gift preferences"}>
            <input
              value={form.gift_preferences}
              onChange={(e) => set("gift_preferences", e.target.value)}
              placeholder={zh ? "重意义轻价格, 有故事的物件" : "Meaning over price, objects with stories"}
              className={inputCls}
            />
          </Field>
          <Field label={zh ? "预算区间" : "Budget range"}>
            <input
              value={form.budget_range}
              onChange={(e) => set("budget_range", e.target.value)}
              placeholder={zh ? "例如：300-1500 元" : "e.g. 300-1500 RMB"}
              className={inputCls}
            />
          </Field>
          <Field label={zh ? "近期情绪状态" : "Recent emotional state"}>
            <input
              value={form.emotional_state}
              onChange={(e) => set("emotional_state", e.target.value)}
              placeholder={zh ? "例如：高强度季度后的重建期" : "e.g. Rebuilding after a demanding quarter"}
              className={inputCls}
            />
          </Field>
          <Field label={zh ? "常见场景" : "Common scenarios"}>
            <input
              value={form.common_scenarios}
              onChange={(e) => set("common_scenarios", e.target.value)}
              placeholder={zh ? "深夜工作, 挚友礼物" : "Late-night work, gifts for close friends"}
              className={inputCls}
            />
          </Field>
        </div>
        {saveButton("preferences", savePreferences)}
      </SectionCard>

      {/* 3 · Relationship profiles */}
      <SectionCard
        step="03"
        title={zh ? "赠予对象档案" : "Recipient profiles"}
        description={
          zh
            ? "为重要的人建立档案，礼物提案会自动带上他们的偏好与重要日期。"
            : "Keep profiles for the people you gift — proposals automatically pick up their preferences and important dates."
        }
      >
        {rels.length > 0 ? (
          <div className="grid gap-3 sm:grid-cols-2">
            {rels.map((r) => (
              <div key={r.id} className="rounded-lg border border-hairline bg-obsidian/60 p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="font-display text-sm text-porcelain">{r.nickname}</p>
                  <Tag>{r.relation_type}</Tag>
                  {r.age_range && <Tag>{r.age_range}</Tag>}
                </div>
                {r.preferences && <p className="mt-2 text-xs leading-relaxed text-mist">{r.preferences}</p>}
                {r.important_dates && (
                  <p className="mt-1 text-xs text-gold/80">
                    {zh ? "重要日期：" : "Important dates: "}
                    {r.important_dates}
                  </p>
                )}
                {r.notes && <p className="mt-1 text-xs italic text-mist">{r.notes}</p>}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-mist">
            {zh
              ? "还没有赠予对象档案。用下方表单添加第一位重要的人。"
              : "No recipient profiles yet. Add the first important person with the form below."}
          </p>
        )}

        <form onSubmit={addRelationship} className="mt-5 rounded-lg border border-hairline bg-obsidian/40 p-4">
          <p className="font-display text-sm text-porcelain">{zh ? "添加赠予对象" : "Add a recipient"}</p>
          <div className="mt-3 grid gap-4 sm:grid-cols-3">
            <Field label={zh ? "关系类型" : "Relation type"}>
              <select
                value={relForm.relation_type}
                onChange={(e) => setRelForm((f) => ({ ...f, relation_type: e.target.value }))}
                className={inputCls}
              >
                {RELATION_TYPES.map((t) => (
                  <option key={t.value} value={t.value}>
                    {zh ? t.zh : t.en}
                  </option>
                ))}
              </select>
            </Field>
            <Field label={zh ? "称呼" : "Nickname"}>
              <input
                value={relForm.nickname}
                onChange={(e) => setRelForm((f) => ({ ...f, nickname: e.target.value }))}
                placeholder={zh ? "例如：阿瑶" : "e.g. A-Yao"}
                className={inputCls}
                maxLength={60}
              />
            </Field>
            <Field label={zh ? "年龄段" : "Age range"}>
              <select
                value={relForm.age_range}
                onChange={(e) => setRelForm((f) => ({ ...f, age_range: e.target.value }))}
                className={inputCls}
              >
                <option value="">{notSet}</option>
                {AGE_RANGES.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <Field label={zh ? "TA 的偏好" : "Their preferences"}>
              <input
                value={relForm.preferences}
                onChange={(e) => setRelForm((f) => ({ ...f, preferences: e.target.value }))}
                placeholder={zh ? "威士忌, 徒步, 胶片摄影" : "Whisky, hiking, film photography"}
                className={inputCls}
                maxLength={400}
              />
            </Field>
            <Field label={zh ? "重要日期" : "Important dates"}>
              <input
                value={relForm.important_dates}
                onChange={(e) => setRelForm((f) => ({ ...f, important_dates: e.target.value }))}
                placeholder={zh ? "生日 09-21, 纪念日 11-02" : "Birthday 09-21, anniversary 11-02"}
                className={inputCls}
                maxLength={200}
              />
            </Field>
          </div>
          <div className="mt-4">
            <Field label={zh ? "备注" : "Notes"}>
              <textarea
                value={relForm.notes}
                onChange={(e) => setRelForm((f) => ({ ...f, notes: e.target.value }))}
                placeholder={zh ? "例如：刚搬到上海，想念成都。" : "e.g. Just moved to Shanghai; misses Chengdu."}
                className={`${inputCls} min-h-[72px] resize-y`}
                maxLength={400}
              />
            </Field>
          </div>
          <div className="mt-4 flex items-center gap-3">
            <Button type="submit" disabled={saving === "relationship"}>
              {saving === "relationship" ? (zh ? "添加中…" : "Adding…") : zh ? "添加档案" : "Add profile"}
            </Button>
            {saved === "relationship" && (
              <span className="text-sm text-jade" role="status">
                ✓ {zh ? "已添加" : "Added"}
              </span>
            )}
          </div>
        </form>
      </SectionCard>

      {/* 4 · Privacy controls */}
      <SectionCard
        step="04"
        title={zh ? "隐私控制" : "Privacy controls"}
        description={
          zh
            ? "记忆、可见性、删除与导出——你的档案由你决定去留。"
            : "Memory, visibility, deletion, and export — your archive stays or goes on your terms."
        }
      >
        <div className="rounded-lg border border-hairline bg-obsidian/60 px-4 py-3">
          <label className="flex cursor-pointer items-start gap-3">
            <input
              type="checkbox"
              checked={form.memory_enabled}
              onChange={(e) => set("memory_enabled", e.target.checked)}
              className="mt-0.5 h-4 w-4 accent-gold"
            />
            <span>
              <span className="text-sm font-medium text-porcelain">
                {zh ? "长期记忆" : "Long-term memory"}
              </span>
              <span className="mt-0.5 block text-xs leading-relaxed text-mist">
                {zh
                  ? "开启后，礼宾会记住你的标签与偏好用于后续生成；关闭后每次对话从零开始，已存档案不受影响。"
                  : "When on, the concierge remembers your tags and preferences for future generations. When off, every conversation starts fresh; saved records are untouched."}
              </span>
            </span>
          </label>
        </div>

        <div className="mt-4">
          <p className="text-xs uppercase tracking-wider text-mist">{zh ? "档案可见性" : "Archive visibility"}</p>
          <div className="mt-2 grid gap-3 sm:grid-cols-3">
            {(
              [
                {
                  value: "private",
                  en: "Private",
                  zh_l: "私密",
                  desc: zh ? "只有你与你的生成可见。" : "Visible only to you and your generations.",
                },
                {
                  value: "co-create",
                  en: "Co-create",
                  zh_l: "共创",
                  desc: zh ? "你加入的共创项目可读取相关偏好。" : "Co-creation projects you join may read relevant preferences.",
                },
                {
                  value: "public",
                  en: "Public",
                  zh_l: "公开",
                  desc: zh ? "公开档案页可展示你选择公开的记录。" : "Public archive pages may display records you mark public.",
                },
              ] as { value: ProfileForm["privacy_level"]; en: string; zh_l: string; desc: string }[]
            ).map((p) => (
              <label
                key={p.value}
                className={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition-colors ${
                  form.privacy_level === p.value ? "border-gold bg-gold/5" : "border-hairline hover:border-gold/40"
                }`}
              >
                <input
                  type="radio"
                  name="privacy_level"
                  value={p.value}
                  checked={form.privacy_level === p.value}
                  onChange={() => set("privacy_level", p.value)}
                  className="mt-0.5 h-4 w-4 accent-gold"
                />
                <span>
                  <span className="text-sm font-medium text-porcelain">{zh ? p.zh_l : p.en}</span>
                  <span className="mt-0.5 block text-xs leading-relaxed text-mist">{p.desc}</span>
                </span>
              </label>
            ))}
          </div>
        </div>

        {saveButton("privacy", savePrivacy)}

        <div className="mt-6 flex flex-col gap-3 border-t border-hairline pt-5 sm:flex-row sm:items-center">
          <Button variant="outline" onClick={exportArchive}>
            {zh ? "导出我的档案" : "Export my archive"}
          </Button>
          {saved === "export" && (
            <span className="text-sm text-jade" role="status">
              ✓ {zh ? "JSON 已下载" : "JSON downloaded"}
            </span>
          )}
          <Button variant="danger" onClick={() => setConfirmDelete(true)}>
            {zh ? "删除档案数据" : "Delete profile data"}
          </Button>
        </div>
        <p className="mt-3 text-xs leading-relaxed text-mist">
          {zh
            ? "导出会生成一份包含标签、偏好、赠予对象、草稿与档案记录的 JSON 文件；删除只移除表达标签与偏好，草稿与档案记录保留。"
            : "Export downloads a JSON file with your tags, preferences, recipients, drafts, and Reserve records. Deletion removes expression tags and preferences only — drafts and Reserve records remain."}
        </p>
      </SectionCard>

      {/* 5 · Generation history */}
      <SectionCard
        step="05"
        title={zh ? "生成历史" : "Generation history"}
        description={
          zh
            ? "最近的礼宾对话摘要。随时回到礼宾继续任何一条线索。"
            : "Summaries of your recent concierge conversations. Return to the concierge to pick up any thread."
        }
      >
        {conversations.length > 0 ? (
          <div className="space-y-3">
            {conversations.map((c) => (
              <div key={c.id} className="rounded-lg border border-hairline bg-obsidian/60 p-4">
                <div className="flex flex-wrap items-center gap-2">
                  <Tag tone="gold">{c.conversation_type}</Tag>
                  <span className="text-xs text-mist">{c.updated_at.slice(0, 10)}</span>
                  <span className="ml-auto text-xs text-mist">
                    {c.token_usage.toLocaleString()} tokens
                  </span>
                </div>
                <p className="mt-2 text-sm leading-relaxed text-porcelain">{c.summary}</p>
              </div>
            ))}
            <Link href="/concierge" className="inline-block text-sm text-gold hover:underline">
              {zh ? "回到礼宾继续 →" : "Continue with the concierge →"}
            </Link>
          </div>
        ) : (
          <div className="rounded-lg border border-hairline bg-obsidian/60 p-5 text-center">
            <p className="text-sm text-mist">
              {zh
                ? "还没有生成记录。和礼宾说一句今天的感受，第一条历史就会出现在这里。"
                : "No generations yet. Tell the concierge one sentence about today, and your first entry appears here."}
            </p>
            <Link href="/concierge" className="mt-2 inline-block text-sm text-gold hover:underline">
              {zh ? "启动 AI 礼宾 →" : "Start the AI concierge →"}
            </Link>
          </div>
        )}
      </SectionCard>

      {/* 6 · Drafts + Reserve */}
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <div className="flex items-center justify-between gap-3">
            <h2 className="font-display text-lg text-porcelain">{zh ? "我的草稿" : "My drafts"}</h2>
            <Link href="/design" className="text-sm text-gold hover:underline">
              {zh ? "全部草稿 →" : "All drafts →"}
            </Link>
          </div>
          <div className="mt-4 space-y-3">
            {drafts.length > 0 ? (
              drafts.slice(0, 4).map((d) => (
                <Link key={d.id} href="/design" className="block rounded-lg border border-hairline bg-obsidian/60 p-4 transition-colors hover:border-gold/40">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="font-display text-sm text-porcelain">{d.title}</p>
                    <StatusPill status={d.status} />
                  </div>
                  <p className="mt-1 text-xs text-mist">
                    {d.object_type} · {d.updated_at.slice(0, 10)}
                    {d.scene ? ` · ${d.scene}` : ""}
                  </p>
                </Link>
              ))
            ) : (
              <p className="text-sm text-mist">
                {zh
                  ? "还没有草稿。在 Forge 或 Supply 生成一个提案并保存，它就会出现在这里。"
                  : "No drafts yet. Generate a proposal in Forge or Supply and save it — it will appear here."}
              </p>
            )}
          </div>
        </Card>
        <Card>
          <div className="flex items-center justify-between gap-3">
            <h2 className="font-display text-lg text-porcelain">{zh ? "档案记录" : "Reserve records"}</h2>
            <Link href="/reserve" className="text-sm text-gold hover:underline">
              {zh ? "进入档案馆 →" : "Enter Reserve →"}
            </Link>
          </div>
          <div className="mt-4 space-y-3">
            {reserve.length > 0 ? (
              reserve.slice(0, 4).map((r) => (
                <Link
                  key={r.id}
                  href={`/reserve/${r.id}`}
                  className="block rounded-lg border border-hairline bg-obsidian/60 p-4 transition-colors hover:border-gold/40"
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="font-display text-sm text-porcelain">{r.object_name}</p>
                    <StatusPill status={r.delivery_status} />
                  </div>
                  <p className="mt-1 text-xs text-mist">
                    {r.zotaix_id} · {r.object_type}
                  </p>
                </Link>
              ))
            ) : (
              <p className="text-sm text-mist">
                {zh
                  ? "还没有档案记录。把一份草稿入档，它会获得 ZOTAIX ID 与证书页。"
                  : "No Reserve records yet. Archive a draft and it receives a ZOTAIX ID and a certificate page."}
              </p>
            )}
          </div>
        </Card>
      </div>

      {/* Delete confirmation modal */}
      {confirmDelete && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-ink/80 p-4"
          role="dialog"
          aria-modal="true"
          aria-label={zh ? "确认删除档案数据" : "Confirm profile data deletion"}
        >
          <div className="zx-card w-full max-w-md p-6">
            <h3 className="font-display text-lg text-porcelain">
              {zh ? "删除档案数据？" : "Delete profile data?"}
            </h3>
            <p className="mt-3 text-sm leading-relaxed text-mist">
              {zh
                ? "这会移除你的自我表达标签与全部偏好。草稿、档案记录与赠予对象档案会保留。此操作立即生效。"
                : "This removes your self-expression tags and all preferences. Drafts, Reserve records, and recipient profiles remain. The action takes effect immediately."}
            </p>
            <div className="mt-5 flex flex-wrap justify-end gap-3">
              <Button variant="ghost" onClick={() => setConfirmDelete(false)} disabled={saving === "delete"}>
                {zh ? "保留我的数据" : "Keep my data"}
              </Button>
              <Button variant="danger" onClick={() => void deleteProfileData()} disabled={saving === "delete"}>
                {saving === "delete" ? (zh ? "删除中…" : "Deleting…") : zh ? "确认删除" : "Delete it"}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
