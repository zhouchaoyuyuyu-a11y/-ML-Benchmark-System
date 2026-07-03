import type { Metadata } from "next";
import { ButtonLink, Section } from "@/components/ui";

export const metadata: Metadata = {
  title: "Offline · ZOTAIX",
  robots: { index: false },
};

export default function OfflinePage() {
  return (
    <Section className="flex min-h-[60vh] flex-col items-center justify-center py-20 text-center">
      <p className="text-xs font-semibold uppercase tracking-[0.3em] text-gold">Offline</p>
      <h1 className="font-display mt-3 text-3xl text-porcelain">The connection paused. Your archive did not.</h1>
      <p className="mt-3 max-w-md text-sm text-mist">
        You appear to be offline. Saved pages remain available; new generations and archive updates will resume once
        your connection returns.
      </p>
      <div className="mt-8">
        <ButtonLink href="/">Retry</ButtonLink>
      </div>
    </Section>
  );
}
