"use client";

import Link from "next/link";
import { useState } from "react";
import type { ChangeEvent, FormEvent, ReactNode } from "react";
import { Button, ButtonLink, Notice, StatusPill } from "@/components/ui";

interface CreateResponse {
  ok?: boolean;
  error?: string;
  note?: string;
  upgradeHint?: string;
  project?: { id: string; title: string; review_status: string };
}

const inputCls =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <div>
      <span className="text-xs uppercase tracking-wider text-mist">{label}</span>
      <div className="mt-1.5">{children}</div>
      {hint && <p className="mt-1 text-xs text-mist">{hint}</p>}
    </div>
  );
}

/** Founder proposal form for the Co-Creation Pool. */
export default function NewProjectClient({
  zh = false,
  signedIn,
  isMember,
}: {
  zh?: boolean;
  signedIn: boolean;
  isMember: boolean;
}) {
  const [title, setTitle] = useState("");
  const [concept, setConcept] = useState("");
  const [productType, setProductType] = useState("wine");
  const [targetQuantity, setTargetQuantity] = useState("50");
  const [tags, setTags] = useState("");
  const [founderQuantity, setFounderQuantity] = useState("2");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [needsAuth, setNeedsAuth] = useState(false);
  const [paywall, setPaywall] = useState<string | null>(null);
  const [success, setSuccess] = useState<{ id: string; title: string; status: string; note: string } | null>(null);

  const onText =
    (setter: (v: string) => void) => (e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) =>
      setter(e.target.value);

  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (title.trim().length < 6) {
      setError(zh ? "标题至少 6 个字符——给项目一个能被记住的名字。" : "The title needs at least 6 characters — give the project a name worth remembering.");
      return;
    }
    if (concept.trim().length < 30) {
      setError(zh ? "概念至少 30 个字符——写清楚它为谁而铸、为什么值得。" : "The concept needs at least 30 characters — say who it is cast for and why it matters.");
      return;
    }
    setLoading(true);
    setError(null);
    setNeedsAuth(false);
    setPaywall(null);
    try {
      const res = await fetch("/api/co-create", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          action: "create",
          title: title.trim(),
          concept: concept.trim(),
          product_type: productType,
          target_quantity: Number(targetQuantity),
          emotion_tags: tags
            .split(/[,，]/)
            .map((t) => t.trim())
            .filter(Boolean)
            .slice(0, 5),
          founder_quantity: Math.min(20, Math.max(1, Number(founderQuantity) || 1)),
        }),
      });
      if (res.status === 401) {
        setNeedsAuth(true);
        return;
      }
      const data = (await res.json().catch(() => ({}))) as CreateResponse;
      if (res.status === 403) {
        setPaywall(data.error ?? null);
        return;
      }
      if (!res.ok || !data.ok || !data.project) {
        setError(data.error ?? (zh ? "提交未成功，请稍后再试。" : "The proposal could not be submitted — please try again."));
        return;
      }
      setSuccess({
        id: data.project.id,
        title: data.project.title,
        status: data.project.review_status || "pending",
        note: data.note ?? "",
      });
    } catch {
      setError(zh ? "网络异常，请稍后再试。" : "A network hiccup interrupted the submission — please try again.");
    } finally {
      setLoading(false);
    }
  }

  if (success) {
    return (
      <div className="zx-card zx-fade-up p-6 sm:p-8">
        <div className="flex flex-wrap items-center gap-2">
          <StatusPill status={success.status} />
          <p className="font-display text-lg text-porcelain">{zh ? "项目已提交评审" : "Project submitted for review"}</p>
        </div>
        <p className="mt-3 text-sm leading-relaxed text-mist">
          {zh
            ? `「${success.title}」已进入平台评审队列。评审通过后项目将公开陈列并开放投票与加入；你的发起人份额与创始版权益已同步登记。`
            : `“${success.title}” has entered the platform review queue. Once approved it goes public with voting and joining open; your founder units and Founder Edition rights are already registered.`}
        </p>
        {success.note && <p className="mt-2 text-xs leading-relaxed text-mist">{success.note}</p>}
        <p className="mt-2 text-xs text-mist">
          {zh ? "项目编号：" : "Project reference: "}
          <span className="text-gold">{success.id}</span>
        </p>
        <div className="mt-5 flex flex-wrap gap-3">
          <ButtonLink href="/profile" variant="gold">
            {zh ? "在个人中心跟踪评审 →" : "Track review in your profile →"}
          </ButtonLink>
          <ButtonLink href="/co-create" variant="outline">
            {zh ? "回到共创池" : "Back to the pool"}
          </ButtonLink>
        </div>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="zx-card space-y-5 p-6 sm:p-8">
      <Field
        label={zh ? "项目标题" : "Project title"}
        hint={zh ? "至少 6 个字符" : "At least 6 characters"}
      >
        <input
          className={inputCls}
          value={title}
          onChange={onText(setTitle)}
          maxLength={140}
          placeholder={zh ? "例：秩序 03:00 —— 给深夜重建者的一瓶" : "e.g. Order 03:00 — a bottle for everyone rebuilding at night"}
        />
      </Field>

      <Field
        label={zh ? "项目概念" : "Concept"}
        hint={zh ? "至少 30 个字符：情绪来源、风味或香氛方向、给谁" : "At least 30 characters: emotional origin, flavor or scent direction, who it is for"}
      >
        <textarea
          className={`${inputCls} min-h-[120px] resize-y`}
          value={concept}
          onChange={onText(setConcept)}
          maxLength={1200}
          placeholder={
            zh
              ? "写下这个对象的故事：它从什么情绪出发，酒体或香氛往哪个方向走，标签想说什么……"
              : "Tell the object's story: which feeling it starts from, where the liquid or scent direction goes, what the label wants to say…"
          }
        />
      </Field>

      <div className="grid gap-5 sm:grid-cols-2">
        <Field label={zh ? "产品类型" : "Product type"}>
          <select className={inputCls} value={productType} onChange={onText(setProductType)}>
            <option value="wine">{zh ? "酒饮" : "Wine / spirit"}</option>
            <option value="fragrance">{zh ? "香氛" : "Fragrance"}</option>
            <option value="bottle">{zh ? "瓶身" : "Bottle"}</option>
            <option value="giftbox">{zh ? "礼盒" : "Gift box"}</option>
          </select>
        </Field>
        <Field
          label={zh ? "目标份数" : "Target quantity"}
          hint={zh ? "目标越高，解锁的定制深度越深" : "Higher targets unlock deeper customization"}
        >
          <select className={inputCls} value={targetQuantity} onChange={onText(setTargetQuantity)}>
            {["50", "100", "300", "500", "1000"].map((q) => (
              <option key={q} value={q}>
                {q} {zh ? "份" : "units"}
              </option>
            ))}
          </select>
        </Field>
      </div>

      <div className="grid gap-5 sm:grid-cols-2">
        <Field
          label={zh ? "情绪标签" : "Emotion tags"}
          hint={zh ? "逗号分隔，最多 5 个" : "Comma-separated, up to 5"}
        >
          <input
            className={inputCls}
            value={tags}
            onChange={onText(setTags)}
            placeholder={zh ? "夜晚, 重建, 秩序" : "night, rebuild, order"}
          />
        </Field>
        <Field
          label={zh ? "发起人首铸份数" : "Founder units"}
          hint={zh ? "你自己预订的份数（1–20）" : "Units you reserve yourself (1–20)"}
        >
          <input
            className={inputCls}
            type="number"
            min={1}
            max={20}
            value={founderQuantity}
            onChange={onText(setFounderQuantity)}
          />
        </Field>
      </div>

      <Button type="submit" variant="gold" disabled={loading} className="w-full sm:w-auto">
        {loading ? (zh ? "提交中…" : "Submitting…") : zh ? "提交项目评审" : "Submit for review"}
      </Button>

      {!signedIn && !needsAuth && (
        <p className="text-xs leading-relaxed text-mist">
          {zh
            ? "提交需要登录账号；发起属于核心序列会员权益。"
            : "Submission requires a signed-in account; founding is a Core Sequence membership benefit."}
        </p>
      )}
      {signedIn && !isMember && !paywall && (
        <p className="text-xs leading-relaxed text-mist">
          {zh
            ? "你的账号目前为免费档：发起项目需要核心序列（Lite / Pro）。你随时可以加入与投票现有项目。"
            : "Your account is currently on the free tier: founding needs the Core Sequence (Lite / Pro). You can always join and vote on existing projects."}
        </p>
      )}

      {needsAuth && (
        <Notice tone="gold" title={zh ? "登录后发起项目" : "Sign in to start a project"}>
          {zh
            ? "项目与发起人权益登记在你的账户下。"
            : "The project and its founder rights are registered under your account."}
          <span className="mt-2 flex flex-wrap gap-3">
            <ButtonLink href="/login" variant="gold" className="!px-4 !py-2">
              {zh ? "去登录" : "Sign in"}
            </ButtonLink>
            <Link href="/register" className="self-center text-xs text-gold hover:underline">
              {zh ? "还没有账号？注册 →" : "New here? Create an account →"}
            </Link>
          </span>
        </Notice>
      )}

      {paywall && (
        <Notice tone="gold" title={zh ? "发起项目是核心序列权益" : "Founding is a Core Sequence benefit"}>
          {paywall}{" "}
          {zh
            ? "加入核心序列即可解锁发起权、私密档案与更高的生成额度。"
            : "Joining the Core Sequence unlocks founding rights, private Reserve records, and a larger generation quota."}
          <span className="mt-2 flex flex-wrap gap-3">
            <ButtonLink href="/membership" variant="gold" className="!px-4 !py-2">
              {zh ? "进入核心序列 →" : "Enter the Core Sequence →"}
            </ButtonLink>
            <Link href="/co-create" className="self-center text-xs text-gold hover:underline">
              {zh ? "先加入现有项目 →" : "Join an existing project instead →"}
            </Link>
          </span>
        </Notice>
      )}

      {error && <Notice tone="ember">{error}</Notice>}
    </form>
  );
}
