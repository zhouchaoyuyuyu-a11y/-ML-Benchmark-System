import { Stat } from "@/components/ui";
import { db } from "@/lib/store";
import CalendarClient from "./CalendarClient";

const corePlatforms = ["WeChat", "Instagram", "TikTok", "X", "YouTube", "LinkedIn"];

export default async function AdminContentCalendarPage() {
  const data = db();
  const items = [...data.content_calendar].sort((a, b) => a.scheduled_at.localeCompare(b.scheduled_at));

  const counts = new Map<string, number>();
  for (const p of corePlatforms) counts.set(p, 0);
  for (const item of items) counts.set(item.platform, (counts.get(item.platform) ?? 0) + 1);

  const published = items.filter((i) => i.status === "published").length;
  const scheduled = items.filter((i) => i.status === "scheduled").length;
  const drafts = items.filter((i) => i.status === "draft").length;

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">Content Calendar</h1>
      <p className="mt-1 text-sm text-mist">
        Editorial pipeline across the WeChat account and the global social matrix — {items.length} items sorted
        by scheduled time. Status moves draft → scheduled → published as posts clear review and go live.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Total items" value={String(items.length)} hint="All platforms" />
        <Stat label="Drafts" value={String(drafts)} hint="In writing or review" />
        <Stat label="Scheduled" value={String(scheduled)} hint="Queued with a publish time" />
        <Stat label="Published" value={String(published)} hint="Live on platform" />
      </div>

      <section className="mt-8">
        <h2 className="font-display text-lg text-porcelain">Platform coverage</h2>
        <p className="mt-1 text-xs text-mist">
          Item count per platform — a zero highlights a channel with nothing in the pipeline.
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          {[...counts.entries()].map(([platform, count]) => (
            <div
              key={platform}
              className={`flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs ${
                count > 0 ? "border-gold/40 text-porcelain" : "border-hairline text-mist"
              }`}
            >
              <span>{platform}</span>
              <span className={`font-mono ${count > 0 ? "text-gold" : "text-mist/70"}`}>{count}</span>
            </div>
          ))}
        </div>
      </section>

      <div className="mt-8 pb-6">
        <CalendarClient items={items} />
      </div>
    </div>
  );
}
