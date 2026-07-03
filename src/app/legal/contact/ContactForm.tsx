"use client";

import { useState } from "react";
import { Button } from "@/components/ui";

const inputClass =
  "w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none";

export default function ContactForm({ zh }: { zh: boolean }) {
  const [name, setName] = useState("");
  const [contact, setContact] = useState("");
  const [topic, setTopic] = useState("");
  const [message, setMessage] = useState("");
  const [state, setState] = useState<"idle" | "sending" | "sent" | "error">("idle");
  const [error, setError] = useState("");

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!contact.trim()) {
      setError(zh ? "请留下邮箱或电话，礼宾才能回复你。" : "Leave an email or phone so the concierge can reply.");
      setState("error");
      return;
    }
    setState("sending");
    setError("");
    try {
      const res = await fetch("/api/trade", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_type: "collaboration",
          name: name.trim() || undefined,
          contact: contact.trim(),
          scenario: topic.trim() || undefined,
          notes: message.trim() ? `contact form · ${message.trim()}` : "contact form",
        }),
      });
      const json = await res.json();
      if (!res.ok || !json.ok) {
        setError(json.error ?? (zh ? "发送失败，请稍后重试或直接写邮件给我们。" : "Sending failed — try again, or write to us by email."));
        setState("error");
        return;
      }
      setState("sent");
    } catch {
      setError(zh ? "网络异常，请稍后重试或直接写邮件给我们。" : "Network issue — try again, or write to us by email.");
      setState("error");
    }
  }

  if (state === "sent") {
    return (
      <div className="rounded-lg border border-jade/40 bg-jade/10 px-4 py-4 text-sm leading-relaxed text-mist">
        <p className="font-medium text-jade">{zh ? "已送达礼宾团队。" : "Delivered to the concierge team."}</p>
        <p className="mt-1">
          {zh
            ? "人工礼宾将在一个工作日内回复你留下的联系方式（工作日 10:00–19:00 CST）。"
            : "A human concierge will reply to the contact you left within one business day (business days, 10:00–19:00 CST)."}
        </p>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-2">
        <input
          className={inputClass}
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={zh ? "你的称呼" : "Your name"}
          aria-label={zh ? "你的称呼" : "Your name"}
        />
        <input
          className={inputClass}
          value={contact}
          onChange={(e) => setContact(e.target.value)}
          placeholder={zh ? "邮箱或电话（必填）" : "Email or phone (required)"}
          aria-label={zh ? "邮箱或电话" : "Email or phone"}
          required
        />
      </div>
      <input
        className={inputClass}
        value={topic}
        onChange={(e) => setTopic(e.target.value)}
        placeholder={zh ? "主题（例如：合作、订单、隐私、举报）" : "Topic (e.g. collaboration, an order, privacy, a report)"}
        aria-label={zh ? "主题" : "Topic"}
      />
      <textarea
        className={`${inputClass} min-h-[110px] resize-y`}
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder={zh ? "想说的话" : "What would you like to tell us?"}
        aria-label={zh ? "留言内容" : "Message"}
      />
      {state === "error" && error && (
        <p className="rounded-md border border-ember/40 bg-ember/10 px-3 py-2 text-xs text-ember">{error}</p>
      )}
      <div className="flex flex-wrap items-center gap-3">
        <Button type="submit" variant="gold" disabled={state === "sending"}>
          {state === "sending" ? (zh ? "发送中…" : "Sending…") : zh ? "发送给礼宾" : "Send to the concierge"}
        </Button>
        <p className="text-xs text-mist">
          {zh ? "提交即同意按照隐私政策处理你的留言。" : "Submitting means your message is handled under the Privacy Policy."}
        </p>
      </div>
    </form>
  );
}
