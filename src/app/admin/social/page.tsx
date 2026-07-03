import { Notice, Stat } from "@/components/ui";
import { db } from "@/lib/store";
import SocialConfigClient from "./SocialConfigClient";

export default async function AdminSocialPage() {
  const data = db();
  const accounts = [...data.social_accounts].sort((a, b) => a.display_order - b.display_order);
  const enabled = accounts.filter((a) => a.enabled).length;
  const withBackup = accounts.filter((a) => (a.backup_url ?? "").trim().length > 0).length;
  const withTracking = accounts.filter((a) => (a.tracking_params ?? "").trim().length > 0).length;

  return (
    <div className="max-w-6xl">
      <h1 className="font-display text-2xl text-porcelain">Social Media Matrix</h1>
      <p className="mt-1 text-sm text-mist">
        The global official-account matrix. Enabled accounts appear on the public /social page and in the site
        footer, ordered by display order; tracking parameters are appended to outbound links so campaign
        attribution stays consistent across platforms.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Platforms" value={String(accounts.length)} hint="Accounts in the matrix" />
        <Stat label="Enabled" value={String(enabled)} hint="Visible on /social and footer" />
        <Stat label="With tracking" value={String(withTracking)} hint="UTM parameters attached" />
        <Stat label="With backup" value={String(withBackup)} hint="Mirror / regional URLs set" />
      </div>

      <div className="mt-6">
        <Notice tone="gold" title="Editing model">
          Each platform row saves independently — change the URL, tracking, order, or visibility and save that
          row. The public page re-reads this table on the next request, so changes are live immediately.
        </Notice>
      </div>

      <div className="mt-6 pb-6">
        <SocialConfigClient accounts={accounts} />
      </div>
    </div>
  );
}
