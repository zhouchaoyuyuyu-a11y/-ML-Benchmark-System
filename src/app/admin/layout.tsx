import type { Metadata } from "next";
import AdminGate from "@/components/admin/AdminGate";
import AdminSidebar from "@/components/admin/AdminSidebar";
import { getSessionUser } from "@/lib/auth";

export const metadata: Metadata = {
  title: "Admin Console · ZOTAIX",
  robots: { index: false, follow: false },
};

export default async function AdminLayout({ children }: { children: React.ReactNode }) {
  const user = await getSessionUser();
  if (!user || user.user_type !== "admin") {
    return <AdminGate />;
  }
  return (
    <div className="flex min-h-screen flex-col lg:flex-row">
      <AdminSidebar nickname={user.nickname} />
      <div className="min-w-0 flex-1 px-4 py-6 sm:px-8">{children}</div>
    </div>
  );
}
