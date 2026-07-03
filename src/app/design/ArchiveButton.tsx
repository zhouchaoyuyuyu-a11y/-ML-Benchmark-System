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

export default function ArchiveButton({ draftId, zh = false }: { draftId: string; zh?: boolean }) {
  const [state, setState] = useState<ArchiveState>("idle");
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
        body: JSON.stringify({ draftId }),
      });
      const data = (await res.json().catch(() => ({}))) as ReserveResponse;
      if (res.status === 401) {
        setState("gate");
        return;
      }
      if (!res.ok || !data.ok || !data.record) {
        setError(data.error ?? (zh ? "入档没有成功，请再试一次。" : "That archive attempt did not go through — try again."));
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
      <span className="inline-flex flex-wrap items-center gap-2 text-xs">
        <span className="inline-flex items-center gap-1.5 rounded-full border border-jade/40 bg-jade/10 px-3 py-1.5 font-medium text-jade">
          ◈ {zh ? "已入档" : "Archived"} · {record.zotaix_id}
        </span>
        <Link href={`/reserve/${record.id}`} className="text-gold hover:underline">
          {zh ? "查看档案证书 →" : "View the Reserve certificate →"}
        </Link>
        {note && <span className="w-full text-mist">{note}</span>}
      </span>
    );
  }

  return (
    <span className="inline-flex flex-wrap items-center gap-2">
      <Button variant="outline" onClick={archive} disabled={state === "loading"}>
        {state === "loading" ? (zh ? "入档中…" : "Archiving…") : zh ? "存入档案馆" : "Archive to Reserve"}
      </Button>
      {state === "gate" && (
        <span className="text-xs text-mist">
          {zh ? "入档需要登录。" : "Archiving requires a signed-in account."}{" "}
          <Link href="/login" className="text-gold hover:underline">
            {zh ? "登录 →" : "Sign in →"}
          </Link>
        </span>
      )}
      {state === "error" && error && <span className="text-xs text-ember">{error}</span>}
    </span>
  );
}
