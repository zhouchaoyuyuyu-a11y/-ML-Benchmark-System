import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { db, newId, now } from "@/lib/store";

/** Admin configuration updates for wechat / social / app / settings / cms / calendar. */
export async function POST(req: Request) {
  const user = await getSessionUser();
  if (!user || user.user_type !== "admin") {
    return NextResponse.json({ ok: false, error: "Admin access required." }, { status: 403 });
  }
  const body = await req.json().catch(() => ({}));
  const section = String(body.section ?? "");
  const database = db();

  if (section === "wechat") {
    const w = database.wechat_config;
    const patch = body.patch ?? {};
    if (typeof patch.official_account_name === "string") w.official_account_name = patch.official_account_name.slice(0, 120);
    if (typeof patch.qr_code_url === "string") w.qr_code_url = patch.qr_code_url.slice(0, 500);
    if (typeof patch.customer_service_url === "string") w.customer_service_url = patch.customer_service_url.slice(0, 500);
    if (typeof patch.enabled === "boolean") w.enabled = patch.enabled;
    if (Array.isArray(patch.menu_config)) w.menu_config = patch.menu_config;
    if (Array.isArray(patch.auto_reply_config)) w.auto_reply_config = patch.auto_reply_config;
    w.updated_at = now();
    return NextResponse.json({ ok: true, wechat: w });
  }

  if (section === "social") {
    const patch = body.patch ?? {};
    const account = database.social_accounts.find((s) => s.id === patch.id);
    if (!account) return NextResponse.json({ ok: false, error: "Account not found." }, { status: 404 });
    if (typeof patch.official_url === "string") account.official_url = patch.official_url.slice(0, 500);
    if (typeof patch.backup_url === "string") account.backup_url = patch.backup_url.slice(0, 500);
    if (typeof patch.tracking_params === "string") account.tracking_params = patch.tracking_params.slice(0, 200);
    if (typeof patch.enabled === "boolean") account.enabled = patch.enabled;
    if (typeof patch.display_order === "number") account.display_order = patch.display_order;
    account.updated_at = now();
    return NextResponse.json({ ok: true, account });
  }

  if (section === "app") {
    const a = database.app_config;
    const patch = body.patch ?? {};
    for (const k of ["ios_download_url", "android_download_url", "apk_download_url", "latest_version", "force_update_version"] as const) {
      if (typeof patch[k] === "string") a[k] = patch[k].slice(0, 500);
    }
    for (const k of ["pwa_enabled", "show_download_banner", "install_prompt_enabled", "downloads_enabled"] as const) {
      if (typeof patch[k] === "boolean") a[k] = patch[k];
    }
    a.updated_at = now();
    return NextResponse.json({ ok: true, app: a });
  }

  if (section === "settings") {
    const s = database.settings;
    const patch = body.patch ?? {};
    const numeric = [
      "guest_daily_chat", "free_daily_chat", "lite_daily_chat", "pro_daily_chat",
      "lite_monthly_proposals", "pro_monthly_proposals",
      "lite_price_month", "lite_price_quarter", "pro_price_month", "pro_price_quarter",
      "co_create_public_threshold", "co_create_review_threshold", "co_create_label_threshold",
      "co_create_flavor_threshold", "co_create_enterprise_threshold", "co_create_supply_threshold",
      "co_create_partner_threshold",
    ] as const;
    for (const k of numeric) {
      if (typeof patch[k] === "number" && patch[k] >= 0) s[k] = patch[k];
    }
    if (typeof patch.brand_line_en === "string") s.brand_line_en = patch.brand_line_en.slice(0, 500);
    if (typeof patch.brand_line_zh === "string") s.brand_line_zh = patch.brand_line_zh.slice(0, 500);
    if (typeof patch.age_gate_enabled === "boolean") s.age_gate_enabled = patch.age_gate_enabled;
    s.updated_at = now();
    return NextResponse.json({ ok: true, settings: s });
  }

  if (section === "cms") {
    const patch = body.patch ?? {};
    const block = database.cms_blocks.find((b) => b.id === patch.id);
    if (!block) return NextResponse.json({ ok: false, error: "Block not found." }, { status: 404 });
    if (typeof patch.content === "string") block.content = patch.content.slice(0, 2000);
    if (typeof patch.enabled === "boolean") block.enabled = patch.enabled;
    block.updated_at = now();
    return NextResponse.json({ ok: true, block });
  }

  if (section === "calendar") {
    const patch = body.patch ?? {};
    if (patch.id) {
      const item = database.content_calendar.find((c) => c.id === patch.id);
      if (!item) return NextResponse.json({ ok: false, error: "Item not found." }, { status: 404 });
      for (const k of ["platform", "title", "content", "media_url", "video_url", "scheduled_at", "owner", "related_url"] as const) {
        if (typeof patch[k] === "string") item[k] = patch[k].slice(0, 1000);
      }
      if (["draft", "scheduled", "published"].includes(patch.status)) item.status = patch.status;
      item.updated_at = now();
      return NextResponse.json({ ok: true, item });
    }
    const item = {
      id: newId("cal"),
      platform: String(patch.platform ?? "Instagram").slice(0, 60),
      title: String(patch.title ?? "Untitled").slice(0, 200),
      content: String(patch.content ?? "").slice(0, 1000),
      media_url: patch.media_url ? String(patch.media_url).slice(0, 500) : undefined,
      video_url: patch.video_url ? String(patch.video_url).slice(0, 500) : undefined,
      scheduled_at: String(patch.scheduled_at ?? new Date().toISOString()),
      status: "draft" as const,
      owner: String(patch.owner ?? user.nickname).slice(0, 80),
      related_url: patch.related_url ? String(patch.related_url).slice(0, 300) : undefined,
      created_at: now(),
      updated_at: now(),
    };
    database.content_calendar.push(item);
    return NextResponse.json({ ok: true, item });
  }

  return NextResponse.json({ ok: false, error: "Unknown section." }, { status: 400 });
}
