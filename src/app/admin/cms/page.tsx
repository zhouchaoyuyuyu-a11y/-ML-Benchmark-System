import { Notice, Stat } from "@/components/ui";
import { db } from "@/lib/store";
import CmsClient from "./CmsClient";

export default async function AdminCmsPage() {
  const data = db();
  const blocks = [...data.cms_blocks].sort((a, b) => a.key.localeCompare(b.key));
  const enabled = blocks.filter((b) => b.enabled).length;
  const pages = new Set(blocks.map((b) => b.page)).size;

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">CMS & Homepage Blocks</h1>
      <p className="mt-1 text-sm text-mist">
        Operational copy that changes more often than the codebase — announcement bars, badges, and service
        notes. Pages look each block up by key; a disabled block simply does not render, so its surface
        collapses cleanly.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Blocks" value={String(blocks.length)} hint="Managed copy slots" />
        <Stat label="Enabled" value={String(enabled)} hint="Currently rendering" />
        <Stat label="Disabled" value={String(blocks.length - enabled)} hint="Hidden from their pages" />
        <Stat label="Pages served" value={String(pages)} hint="Distinct consuming routes" />
      </div>

      <div className="mt-6">
        <Notice tone="gold" title="How blocks are consumed">
          Each server page calls db().cms_blocks and finds its block by key with enabled=true — for example the
          homepage reads home.announcement for the bar above the hero. Content is plain text, saved per block,
          and live on the next request.
        </Notice>
      </div>

      <div className="mt-6 pb-6">
        <CmsClient blocks={blocks} />
      </div>
    </div>
  );
}
