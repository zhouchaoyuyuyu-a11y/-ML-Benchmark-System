"use client";

import Link from "next/link";
import { useState } from "react";
import { ButtonLink, Card, EmptyState, ProgressBar, Tag } from "@/components/ui";
import type { CoCreationProject } from "@/lib/types";

interface VoteResponse {
  ok?: boolean;
  error?: string;
  votes?: number;
}

/** Public co-creation project cards with an optimistic vote action. */
export default function CoCreateListClient({
  zh = false,
  projects,
}: {
  zh?: boolean;
  projects: CoCreationProject[];
}) {
  const [votes, setVotes] = useState<Record<string, number>>(() =>
    Object.fromEntries(projects.map((p) => [p.id, p.votes]))
  );
  const [pending, setPending] = useState<Record<string, boolean>>({});
  const [errors, setErrors] = useState<Record<string, string | null>>({});

  async function vote(projectId: string) {
    if (pending[projectId]) return;
    setPending((s) => ({ ...s, [projectId]: true }));
    setErrors((s) => ({ ...s, [projectId]: null }));
    // Optimistic +1; reconcile with the server count on success, revert on failure.
    setVotes((s) => ({ ...s, [projectId]: (s[projectId] ?? 0) + 1 }));
    try {
      const res = await fetch("/api/co-create", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action: "vote", projectId }),
      });
      const data = (await res.json().catch(() => ({}))) as VoteResponse;
      if (!res.ok || !data.ok) {
        setVotes((s) => ({ ...s, [projectId]: Math.max(0, (s[projectId] ?? 1) - 1) }));
        setErrors((s) => ({
          ...s,
          [projectId]: data.error ?? (zh ? "投票没有成功，请再试一次。" : "That vote did not land — please try again."),
        }));
        return;
      }
      if (typeof data.votes === "number") {
        setVotes((s) => ({ ...s, [projectId]: data.votes as number }));
      }
    } catch {
      setVotes((s) => ({ ...s, [projectId]: Math.max(0, (s[projectId] ?? 1) - 1) }));
      setErrors((s) => ({
        ...s,
        [projectId]: zh ? "网络波动了一下，请再试一次。" : "The connection wobbled — please try again.",
      }));
    } finally {
      setPending((s) => ({ ...s, [projectId]: false }));
    }
  }

  if (projects.length === 0) {
    return (
      <EmptyState
        title={zh ? "共创池正在等待下一个概念" : "The pool is waiting for its next concept"}
        description={
          zh
            ? "成为第一个发起者：把一个 AI 生成的对象带进共创池，集结你的支持者。"
            : "Be the first founder: bring an AI-generated object into the pool and gather your supporters."
        }
        action={
          <ButtonLink href="/co-create/new" variant="gold">
            {zh ? "发起一个项目" : "Start a project"}
          </ButtonLink>
        }
      />
    );
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {projects.map((p) => (
        <Card key={p.id} hover className="flex h-full flex-col">
          <div className="flex flex-wrap items-center gap-2">
            <Tag tone="gold">{p.product_type}</Tag>
            {p.emotion_tags.map((t) => (
              <Tag key={t}>{t}</Tag>
            ))}
          </div>
          <Link href={`/co-create/${p.id}`} className="mt-3 block">
            <p className="font-display text-lg text-porcelain transition-colors hover:text-gold">{p.title}</p>
          </Link>
          <p className="mt-2 line-clamp-2 text-sm leading-relaxed text-mist">{p.concept}</p>
          <div className="mt-4 space-y-1.5">
            <div className="flex justify-between text-xs text-mist">
              <span>
                {p.current_quantity}/{p.target_quantity} {zh ? "已预订" : "reserved"}
              </span>
              <span>
                {p.supporters} {zh ? "位支持者" : "supporters"}
              </span>
            </div>
            <ProgressBar value={p.current_quantity} max={p.target_quantity} />
          </div>
          <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
            <button
              type="button"
              onClick={() => vote(p.id)}
              disabled={!!pending[p.id]}
              className="inline-flex items-center gap-2 rounded-md border border-hairline px-4 py-2 text-sm font-medium text-porcelain transition-colors hover:border-gold hover:text-gold disabled:cursor-not-allowed disabled:opacity-50"
            >
              <span className="text-gold">▲</span>
              {zh ? "投票" : "Vote"} · {votes[p.id] ?? p.votes}
            </button>
            <Link href={`/co-create/${p.id}`} className="text-sm text-gold hover:underline">
              {zh ? "查看详情并加入 →" : "View & join →"}
            </Link>
          </div>
          {errors[p.id] && <p className="mt-2 text-xs text-ember">{errors[p.id]}</p>}
        </Card>
      ))}
    </div>
  );
}
