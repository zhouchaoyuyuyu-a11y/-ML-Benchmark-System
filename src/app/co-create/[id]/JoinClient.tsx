"use client";

import Link from "next/link";
import { useState } from "react";
import { Button, ButtonLink, Notice, ProgressBar, StatusPill } from "@/components/ui";

interface JoinResponse {
  ok?: boolean;
  error?: string;
  project?: { current_quantity: number; supporters: number };
}

interface VoteResponse {
  ok?: boolean;
  error?: string;
  votes?: number;
}

const stepBtnCls =
  "inline-flex h-9 w-9 items-center justify-center rounded-md border border-hairline text-base text-porcelain transition-colors hover:border-gold hover:text-gold disabled:cursor-not-allowed disabled:opacity-40";

/** Join stepper, vote button, and live numbers for a co-creation project. */
export default function JoinClient({
  zh = false,
  projectId,
  gathering,
  targetQuantity,
  initialQuantity,
  initialSupporters,
  initialVotes,
  membersCount,
}: {
  zh?: boolean;
  projectId: string;
  gathering: boolean;
  targetQuantity: number;
  initialQuantity: number;
  initialSupporters: number;
  initialVotes: number;
  membersCount: number;
}) {
  const [quantity, setQuantity] = useState(1);
  const [current, setCurrent] = useState(initialQuantity);
  const [supporters, setSupporters] = useState(initialSupporters);
  const [votes, setVotes] = useState(initialVotes);
  const [joining, setJoining] = useState(false);
  const [voting, setVoting] = useState(false);
  const [needsAuth, setNeedsAuth] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [joined, setJoined] = useState<number | null>(null);

  async function join() {
    setJoining(true);
    setError(null);
    try {
      const res = await fetch("/api/co-create", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action: "join", projectId, quantity }),
      });
      if (res.status === 401) {
        setNeedsAuth(true);
        return;
      }
      const data = (await res.json().catch(() => ({}))) as JoinResponse;
      if (!res.ok || !data.ok || !data.project) {
        setError(data.error ?? (zh ? "加入没有成功，请再试一次。" : "Joining did not go through — please try again."));
        return;
      }
      setCurrent(data.project.current_quantity);
      setSupporters(data.project.supporters);
      setJoined((prev) => (prev ?? 0) + quantity);
    } catch {
      setError(zh ? "网络波动了一下，请再试一次。" : "The connection wobbled — please try again.");
    } finally {
      setJoining(false);
    }
  }

  async function vote() {
    if (voting) return;
    setVoting(true);
    setError(null);
    setVotes((v) => v + 1); // optimistic
    try {
      const res = await fetch("/api/co-create", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action: "vote", projectId }),
      });
      const data = (await res.json().catch(() => ({}))) as VoteResponse;
      if (!res.ok || !data.ok) {
        setVotes((v) => Math.max(0, v - 1));
        setError(data.error ?? (zh ? "投票没有成功，请再试一次。" : "That vote did not land — please try again."));
        return;
      }
      if (typeof data.votes === "number") setVotes(data.votes);
    } catch {
      setVotes((v) => Math.max(0, v - 1));
      setError(zh ? "网络波动了一下，请再试一次。" : "The connection wobbled — please try again.");
    } finally {
      setVoting(false);
    }
  }

  return (
    <div className="zx-card p-5 sm:p-6">
      <p className="font-display text-base text-porcelain">{zh ? "集结进度" : "Gathering progress"}</p>

      <div className="mt-4 space-y-1.5">
        <div className="flex justify-between text-xs text-mist">
          <span>
            {current}/{targetQuantity} {zh ? "已预订" : "reserved"}
          </span>
          <span>
            {Math.min(100, Math.round((current / Math.max(1, targetQuantity)) * 100))}%
          </span>
        </div>
        <ProgressBar value={current} max={targetQuantity} />
      </div>

      <div className="mt-4 grid grid-cols-3 gap-2 text-center">
        <div className="rounded-md border border-hairline px-2 py-2.5">
          <p className="font-display text-lg text-porcelain">{supporters}</p>
          <p className="text-[11px] uppercase tracking-wider text-mist">{zh ? "支持者" : "supporters"}</p>
        </div>
        <div className="rounded-md border border-hairline px-2 py-2.5">
          <p className="font-display text-lg text-porcelain">{membersCount}</p>
          <p className="text-[11px] uppercase tracking-wider text-mist">{zh ? "成员" : "members"}</p>
        </div>
        <div className="rounded-md border border-hairline px-2 py-2.5">
          <p className="font-display text-lg text-porcelain">{votes}</p>
          <p className="text-[11px] uppercase tracking-wider text-mist">{zh ? "票数" : "votes"}</p>
        </div>
      </div>

      {gathering ? (
        <div className="mt-5">
          <p className="text-xs uppercase tracking-wider text-mist">{zh ? "预订份数（1–20）" : "Units to reserve (1–20)"}</p>
          <div className="mt-2 flex items-center gap-3">
            <button
              type="button"
              className={stepBtnCls}
              onClick={() => setQuantity((q) => Math.max(1, q - 1))}
              disabled={quantity <= 1 || joining}
              aria-label={zh ? "减少份数" : "Decrease quantity"}
            >
              −
            </button>
            <span className="font-display min-w-[3rem] text-center text-2xl text-porcelain">{quantity}</span>
            <button
              type="button"
              className={stepBtnCls}
              onClick={() => setQuantity((q) => Math.min(20, q + 1))}
              disabled={quantity >= 20 || joining}
              aria-label={zh ? "增加份数" : "Increase quantity"}
            >
              +
            </button>
          </div>
          <div className="mt-4 flex flex-col gap-2.5">
            <Button variant="gold" onClick={join} disabled={joining} className="w-full">
              {joining ? (zh ? "加入中…" : "Joining…") : zh ? `加入共创 · ${quantity} 份` : `Join with ${quantity} unit${quantity > 1 ? "s" : ""}`}
            </Button>
            <Button variant="outline" onClick={vote} disabled={voting} className="w-full">
              <span className="text-gold">▲</span> {zh ? "为项目投票" : "Vote for this project"} · {votes}
            </Button>
          </div>
          <p className="mt-3 text-xs leading-relaxed text-mist">
            {zh
              ? "预订不收款：人工礼宾确认与合规审核之后，才会进入付款环节。"
              : "Reserving takes no payment — the payment step only begins after human concierge confirmation and compliance review."}
          </p>
        </div>
      ) : (
        <div className="mt-5">
          <Button variant="outline" onClick={vote} disabled={voting} className="w-full">
            <span className="text-gold">▲</span> {zh ? "为项目投票" : "Vote for this project"} · {votes}
          </Button>
          <p className="mt-3 text-xs leading-relaxed text-mist">
            {zh
              ? "这个项目当前不在集结阶段；投票依然帮助它进入下一轮评审。"
              : "This project is not gathering right now; votes still help it into the next review round."}
          </p>
        </div>
      )}

      {joined !== null && (
        <div className="zx-fade-up mt-4 rounded-lg border border-jade/30 bg-jade/5 p-4">
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill status="active" />
            <p className="text-sm font-medium text-porcelain">
              {zh ? `已预订 ${joined} 份` : `${joined} unit${joined > 1 ? "s" : ""} reserved`}
            </p>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-mist">
            {zh
              ? "你的份额已计入门槛进度。交付后，它会出现在你的 Reserve 档案馆。"
              : "Your share now counts toward the thresholds. After delivery it appears in your Reserve archive."}
          </p>
          <Link href="/reserve" className="mt-2 inline-block text-xs text-gold hover:underline">
            {zh ? "打开档案馆 →" : "Open the Reserve →"}
          </Link>
        </div>
      )}

      {needsAuth && (
        <div className="mt-4">
          <Notice tone="gold" title={zh ? "注册后加入共创" : "Register to join the casting"}>
            {zh
              ? "加入需要一个账号，这样你的份额、数字印记与档案才有归属。"
              : "Joining needs an account so your share, digital mark, and archive have somewhere to live."}
            <span className="mt-2 flex flex-wrap gap-3">
              <ButtonLink href="/register" variant="gold" className="!px-4 !py-2">
                {zh ? "注册账号" : "Create an account"}
              </ButtonLink>
              <Link href="/login" className="self-center text-xs text-gold hover:underline">
                {zh ? "已有账号？登录 →" : "Already have one? Sign in →"}
              </Link>
            </span>
          </Notice>
        </div>
      )}

      {error && (
        <div className="mt-4">
          <Notice tone="ember">{error}</Notice>
        </div>
      )}
    </div>
  );
}
