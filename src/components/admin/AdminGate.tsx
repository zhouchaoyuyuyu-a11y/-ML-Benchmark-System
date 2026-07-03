"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

/** Sign-in card shown when a non-admin hits /admin. */
export default function AdminGate() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const data = await res.json();
    setLoading(false);
    if (!data.ok) {
      setError(data.error ?? "Sign-in failed.");
      return;
    }
    if (data.user?.user_type !== "admin") {
      setError("This account does not have admin access.");
      return;
    }
    router.refresh();
  }

  return (
    <div className="flex min-h-[70vh] items-center justify-center px-4">
      <div className="zx-card w-full max-w-md p-6">
        <p className="font-display text-lg tracking-[0.2em] text-porcelain">ZOTAIX ADMIN</p>
        <p className="mt-2 text-sm text-mist">Operations console — admin accounts only.</p>
        <form onSubmit={submit} className="mt-6 space-y-3">
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Admin email"
            className="w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none"
          />
          <input
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            className="w-full rounded-md border border-hairline bg-ink px-3 py-2.5 text-sm text-porcelain placeholder:text-mist focus:border-gold focus:outline-none"
          />
          {error && <p className="text-xs text-ember">{error}</p>}
          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-md bg-gold px-4 py-2.5 text-sm font-medium text-ink transition-colors hover:bg-gold-deep hover:text-porcelain disabled:opacity-50"
          >
            {loading ? "Signing in…" : "Enter console"}
          </button>
        </form>
        <p className="mt-4 text-xs leading-relaxed text-mist">
          Demo environment: <code className="text-gold">admin@zotaix.demo</code> / <code className="text-gold">zotaix-demo</code>.
          In production, set ADMIN_EMAIL and real credentials via environment variables.
        </p>
      </div>
    </div>
  );
}
