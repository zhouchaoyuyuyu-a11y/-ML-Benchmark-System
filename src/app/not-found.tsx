import { ButtonLink, Section } from "@/components/ui";

export default function NotFound() {
  return (
    <Section className="flex min-h-[60vh] flex-col items-center justify-center py-20 text-center">
      <p className="text-xs font-semibold uppercase tracking-[0.3em] text-gold">404</p>
      <h1 className="font-display mt-3 text-3xl text-porcelain">This page is outside the current order.</h1>
      <p className="mt-3 max-w-md text-sm text-mist">
        The address may have moved into the archive. Return to the platform, or ask the concierge to help you find what you were looking for.
      </p>
      <div className="mt-8 flex gap-3">
        <ButtonLink href="/">Back to ZOTAIX</ButtonLink>
        <ButtonLink href="/concierge" variant="outline">Ask the concierge</ButtonLink>
      </div>
    </Section>
  );
}
