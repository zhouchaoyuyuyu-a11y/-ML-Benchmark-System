import Link from "next/link";
import type { ReactNode } from "react";

/* Shared UI primitives. Server-safe (no hooks) so they can be used from both
   server and client components. */

export function Section({ children, className = "", id }: { children: ReactNode; className?: string; id?: string }) {
  return (
    <section id={id} className={`mx-auto w-full max-w-6xl px-4 sm:px-6 ${className}`}>
      {children}
    </section>
  );
}

export function SectionHeader({
  eyebrow,
  title,
  description,
  className = "",
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  className?: string;
}) {
  return (
    <div className={`max-w-3xl ${className}`}>
      {eyebrow && (
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-gold">{eyebrow}</p>
      )}
      <h2 className="font-display text-2xl text-porcelain sm:text-3xl">{title}</h2>
      {description && <p className="mt-3 text-sm leading-relaxed text-mist sm:text-base">{description}</p>}
    </div>
  );
}

export function PageHero({
  eyebrow,
  title,
  description,
  children,
  tone = "default",
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  children?: ReactNode;
  tone?: "default" | "supply" | "maison";
}) {
  const accent = tone === "supply" ? "text-supply" : "text-gold";
  return (
    <div className="zx-grid-bg border-b border-hairline">
      <Section className="py-14 sm:py-20">
        {eyebrow && (
          <p className={`mb-3 text-xs font-semibold uppercase tracking-[0.25em] ${accent}`}>{eyebrow}</p>
        )}
        <h1 className="font-display max-w-4xl text-3xl leading-tight text-porcelain sm:text-5xl">{title}</h1>
        {description && (
          <p className="mt-5 max-w-3xl text-sm leading-relaxed text-mist sm:text-lg">{description}</p>
        )}
        {children && <div className="mt-8 flex flex-wrap gap-3">{children}</div>}
      </Section>
    </div>
  );
}

type ButtonVariant = "gold" | "outline" | "ghost" | "supply" | "danger";

const buttonStyles: Record<ButtonVariant, string> = {
  gold: "bg-gold text-ink hover:bg-gold-deep hover:text-porcelain border border-gold",
  outline: "border border-hairline text-porcelain hover:border-gold hover:text-gold",
  ghost: "text-mist hover:text-porcelain",
  supply: "bg-supply/90 text-ink hover:bg-supply border border-supply",
  danger: "border border-ember/60 text-ember hover:bg-ember/10",
};

export function ButtonLink({
  href,
  children,
  variant = "gold",
  className = "",
}: {
  href: string;
  children: ReactNode;
  variant?: ButtonVariant;
  className?: string;
}) {
  return (
    <Link
      href={href}
      className={`inline-flex items-center justify-center gap-2 rounded-md px-5 py-2.5 text-sm font-medium transition-colors ${buttonStyles[variant]} ${className}`}
    >
      {children}
    </Link>
  );
}

export function Button({
  children,
  variant = "gold",
  className = "",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant }) {
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-md px-5 py-2.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${buttonStyles[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function Card({ children, className = "", hover = false }: { children: ReactNode; className?: string; hover?: boolean }) {
  return <div className={`zx-card ${hover ? "zx-card-hover" : ""} p-5 sm:p-6 ${className}`}>{children}</div>;
}

export function Tag({ children, tone = "default" }: { children: ReactNode; tone?: "default" | "gold" | "supply" | "jade" | "ember" }) {
  const tones = {
    default: "border-hairline text-mist",
    gold: "border-gold/40 text-gold",
    supply: "border-supply/40 text-supply",
    jade: "border-jade/40 text-jade",
    ember: "border-ember/40 text-ember",
  };
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs ${tones[tone]}`}>{children}</span>
  );
}

const statusTones: Record<string, string> = {
  approved: "text-jade border-jade/40 bg-jade/10",
  passed: "text-jade border-jade/40 bg-jade/10",
  paid: "text-jade border-jade/40 bg-jade/10",
  published: "text-jade border-jade/40 bg-jade/10",
  delivered: "text-jade border-jade/40 bg-jade/10",
  active: "text-jade border-jade/40 bg-jade/10",
  sent: "text-jade border-jade/40 bg-jade/10",
  pending: "text-gold border-gold/40 bg-gold/10",
  gathering: "text-gold border-gold/40 bg-gold/10",
  scheduled: "text-gold border-gold/40 bg-gold/10",
  drafting: "text-gold border-gold/40 bg-gold/10",
  review: "text-gold border-gold/40 bg-gold/10",
  in_production: "text-gold border-gold/40 bg-gold/10",
  awaiting_concierge: "text-gold border-gold/40 bg-gold/10",
  test_mode: "text-supply border-supply/40 bg-supply/10",
  draft: "text-mist border-hairline bg-veil",
  digital: "text-supply border-supply/40 bg-supply/10",
  rejected: "text-ember border-ember/40 bg-ember/10",
  flagged: "text-ember border-ember/40 bg-ember/10",
  escalated: "text-ember border-ember/40 bg-ember/10",
  revision: "text-ember border-ember/40 bg-ember/10",
};

export function StatusPill({ status }: { status: string }) {
  const tone = statusTones[status] ?? "text-mist border-hairline bg-veil";
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${tone}`}>
      {status.replace(/_/g, " ")}
    </span>
  );
}

export function Notice({
  children,
  tone = "gold",
  title,
}: {
  children: ReactNode;
  tone?: "gold" | "ember" | "supply";
  title?: string;
}) {
  const tones = {
    gold: "border-gold/30 bg-gold/5",
    ember: "border-ember/30 bg-ember/5",
    supply: "border-supply/30 bg-supply/5",
  };
  return (
    <div className={`rounded-lg border px-4 py-3 text-sm leading-relaxed text-mist ${tones[tone]}`}>
      {title && <p className="mb-1 font-medium text-porcelain">{title}</p>}
      {children}
    </div>
  );
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="zx-card flex flex-col items-center gap-3 px-6 py-12 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-full border border-hairline text-gold">◈</div>
      <p className="font-display text-lg text-porcelain">{title}</p>
      <p className="max-w-md text-sm text-mist">{description}</p>
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="zx-card p-4">
      <p className="text-xs uppercase tracking-wider text-mist">{label}</p>
      <p className="font-display mt-1 text-2xl text-porcelain">{value}</p>
      {hint && <p className="mt-1 text-xs text-mist">{hint}</p>}
    </div>
  );
}

export function Meridian({ className = "" }: { className?: string }) {
  return <div className={`zx-meridian ${className}`} />;
}

export function ProgressBar({ value, max, tone = "gold" }: { value: number; max: number; tone?: "gold" | "supply" }) {
  const pct = Math.min(100, Math.round((value / Math.max(1, max)) * 100));
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-veil">
      <div
        className={tone === "gold" ? "h-full rounded-full bg-gold" : "h-full rounded-full bg-supply"}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function DefinitionRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-1 gap-1 border-b border-hairline py-3 last:border-0 sm:grid-cols-[200px_1fr] sm:gap-4">
      <dt className="text-xs uppercase tracking-wider text-mist sm:pt-0.5">{label}</dt>
      <dd className="text-sm text-porcelain">{children}</dd>
    </div>
  );
}
