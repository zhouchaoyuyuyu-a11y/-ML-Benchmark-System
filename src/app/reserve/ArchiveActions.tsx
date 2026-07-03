"use client";

import Link from "next/link";
import { useState } from "react";
import { Button } from "@/components/ui";

interface ReserveResponse {
  ok?: boolean;
  error?: string;
  note?: string;
  record?: { id: string; zotaix_id: string };
}

type ArchiveState = "idle" | "loading" | "done" | "gate" | "error";

/** Archive a saved draft into the Reserve as a lifetime record. */
export default function ArchiveActions({ draftId, zh = false }: { draftId: string; zh?: boolean }) {
  const [state, setState] = useState<ArchiveState>("idle");
  const [asPrivate, setAsPrivate] = useState(false);
  const [record, setRecord] = useState<{ id: string; zotaix_id: string } | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function archive() {
    setState("loading");
    setError(null);
    try {
      const res = await fetch("/api/reserve", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ draftId, privacy_level: asPrivate ? "private" : "public" }),
      });
      if (res.status === 401) {
        setState("gate");
        return;
      }
      const data = (await res.json().catch(() => ({}))) as ReserveResponse;
      if (!res.ok || !data.ok || !data.record) {
        setError(data.error ?? (zh ? "入档没有成功，请再试一次。" : "That archive attempt did not go through — please try again."));
        setState("error");
        return;
      }
      setRecord(data.record);
      setNote(data.note ?? null);
      setState("done");
    } catch {
      setError(zh ? "网络波动了一下，请再试一次。" : "The connection wobbled — please try again.");
      setState("error");
    }
  }

  if (state === "done" && record) {
    return (
      <div className="zx-fade-up rounded-lg border border-jade/30 bg-jade/5 p-4">
        <p className="text-sm font-medium text-jade">
          ◈ {zh ? "已入档" : "Archived"} · <span className="text-porcelain">{record.zotaix_id}</span>
        </p>
        <p className="mt-1.5 text-xs leading-relaxed text-mist">
          {zh
            ? "这个对象现在拥有终身编号、QR/NFC 标识与证书页面。"
            : "This object now carries a lifetime serial, a QR/NFC identity, and a certificate page."}
        </p>
        {note && <p className="mt-1.5 text-xs leading-relaxed text-mist">{note}</p>}
        <Link href={`/reserve/${record.id}`} className="mt-2 inline-block text-sm text-gold hover:underline">
          {zh ? "打开证书页 →" : "Open the certificate →"}
        </Link>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-3">
      <Button variant="gold" onClick={archive} disabled={state === "loading"}>
        {state === "loading" ? (zh ? "入档中…" : "Archiving…") : zh ? "入档为 Reserve 记录" : "Archive to Reserve"}
      </Button>
      <label className="flex cursor-pointer items-center gap-2 text-xs text-mist">
        <input
          type="checkbox"
          checked={asPrivate}
          onChange={(e) => setAsPrivate(e.target.checked)}
          className="h-3.5 w-3.5 accent-[#c8a962]"
        />
        {zh ? "封存为私密（核心序列权益）" : "Seal as private (Core Sequence benefit)"}
      </label>
      {state === "gate" && (
        <span className="w-full text-xs text-mist">
          {zh ? "入档需要登录账号。" : "Archiving requires a signed-in account."}{" "}
          <Link href="/login" className="text-gold hover:underline">
            {zh ? "去登录 →" : "Sign in →"}
          </Link>
        </span>
      )}
      {state === "error" && error && <span className="w-full text-xs text-ember">{error}</span>}
    </div>
  );
}
