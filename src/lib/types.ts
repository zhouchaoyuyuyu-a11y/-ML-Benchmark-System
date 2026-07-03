// ZOTAIX data model. Mirrors db/schema.sql (Supabase/Postgres-compatible).
// The runtime store (src/lib/store.ts) uses these types with a seeded
// in-memory fallback when DATABASE_URL is not configured.

export type Locale = "en" | "zh";

export type UserType = "guest" | "registered" | "member" | "enterprise" | "admin";
export type MembershipLevel = "free" | "lite" | "pro" | "enterprise";

export interface User {
  id: string;
  phone?: string;
  email: string;
  password_hash?: string;
  nickname: string;
  avatar?: string;
  user_type: UserType;
  membership_level: MembershipLevel;
  daily_quota: number;
  used_quota: number;
  created_at: string;
  updated_at: string;
}

export interface UserProfile {
  id: string;
  user_id: string;
  mbti?: string;
  zodiac?: string;
  blood_type?: string;
  age_range?: string;
  nickname?: string;
  address_style?: string;
  favorite_colors?: string[];
  scent_preferences?: string[];
  alcohol_preferences?: string[];
  alcohol_tolerance?: string;
  non_alcohol_ok?: boolean;
  music?: string;
  movies?: string;
  cities?: string;
  literary_imagery?: string;
  visual_preferences?: string[];
  gift_preferences?: string[];
  budget_range?: string;
  emotional_state?: string;
  common_scenarios?: string[];
  personality_tags?: string[];
  privacy_level: "private" | "co-create" | "public";
  memory_enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface RelationshipProfile {
  id: string;
  user_id: string;
  relation_type: string;
  nickname: string;
  age_range?: string;
  preferences?: string;
  important_dates?: string;
  notes?: string;
  privacy_level: "private" | "public";
  created_at: string;
  updated_at: string;
}

export type ConversationType = "daily" | "gift" | "product" | "fragrance" | "wine" | "enterprise";

export interface Conversation {
  id: string;
  user_id?: string;
  visitor_id?: string;
  mode: "guest" | "registered" | "member";
  conversation_type: ConversationType;
  summary: string;
  token_usage: number;
  created_at: string;
  updated_at: string;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  structured?: ConceptProposal | null;
  token_usage: number;
  created_at: string;
}

export interface AiUsageLog {
  id: string;
  user_id?: string;
  visitor_id?: string;
  model: string;
  action_type: "chat" | "proposal" | "creative" | "image" | "export";
  tokens_used: number;
  cost_estimate: number;
  quota_consumed: number;
  created_at: string;
}

export type ObjectType = "spirit" | "fragrance" | "bottle" | "giftbox" | "label" | "enterprise_gift";
export type DraftStatus = "draft" | "saved" | "ordered" | "reviewed";

export interface ObjectDraft {
  id: string;
  user_id: string;
  object_type: ObjectType;
  title: string;
  scene?: string;
  recipient?: string;
  budget?: string;
  emotion_tags: string[];
  liquid_direction?: string;
  scent_direction?: string;
  label_copy?: string;
  visual_style?: string;
  names?: string[];
  status: DraftStatus;
  public_visible: boolean;
  created_at: string;
  updated_at: string;
}

export interface DesignVersion {
  id: string;
  object_draft_id: string;
  version_name: string;
  design_payload: Record<string, string>;
  image_url?: string;
  model_url?: string;
  version_hash: string;
  created_at: string;
}

export interface Membership {
  id: string;
  user_id: string;
  plan: MembershipLevel;
  monthly_quota: number;
  daily_chat_limit: number;
  premium_generation_limit: number;
  image_generation_limit: number;
  export_enabled: boolean;
  reserve_enabled: boolean;
  concierge_enabled: boolean;
  started_at: string;
  expires_at: string;
}

export type ReviewStatus = "pending" | "approved" | "rejected" | "revision" | "escalated";

export interface CoCreationProject {
  id: string;
  creator_user_id: string;
  title: string;
  concept: string;
  product_type: "wine" | "fragrance" | "bottle" | "giftbox";
  target_quantity: number;
  current_quantity: number;
  supporters: number;
  status: "gathering" | "review" | "production" | "delivered" | "closed";
  founder_benefit: string;
  public_visible: boolean;
  review_status: ReviewStatus;
  emotion_tags: string[];
  votes: number;
  created_at: string;
  updated_at: string;
}

export interface CoCreationMember {
  id: string;
  project_id: string;
  user_id: string;
  role: "founder" | "participant";
  quantity: number;
  payment_status: "unpaid" | "reserved" | "paid" | "refunded";
  joined_at: string;
}

export interface ReserveRecord {
  id: string;
  user_id: string;
  object_draft_id?: string;
  zotaix_id: string;
  object_type: ObjectType | "badge" | "co_creation" | "design_version";
  object_name: string;
  design_version_id?: string;
  emotion_tags: string[];
  relationship_scene?: string;
  product_direction?: string;
  label_copy?: string;
  scent_direction?: string;
  liquid_direction?: string;
  visual_style?: string;
  batch_id?: string;
  qr_nfc_id: string;
  certificate_url?: string;
  privacy_level: "private" | "public";
  co_create_eligible: boolean;
  delivery_status: "digital" | "pending_review" | "in_production" | "delivered";
  repurchase_eligible: boolean;
  aftercare_status: "active" | "expired" | "none";
  created_at: string;
  updated_at: string;
}

export interface TradeRequest {
  id: string;
  user_id: string;
  object_draft_id?: string;
  request_type: "quote" | "authorization" | "enterprise" | "collaboration" | "replenishment";
  organization?: string;
  contact?: string;
  quantity: number;
  budget: string;
  deadline?: string;
  delivery_region?: string;
  liquid_direction?: string;
  scent_direction?: string;
  bottle_direction?: string;
  packaging_direction?: string;
  sample_path?: string;
  invoice_required: boolean;
  logistics_notes?: string;
  compliance_status: "unchecked" | "passed" | "flagged";
  human_review_status: ReviewStatus;
  quote_status: "none" | "drafting" | "sent" | "accepted" | "declined";
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface ModerationLog {
  id: string;
  user_id?: string;
  object_id: string;
  content_type: string;
  risk_type:
    | "sensitive"
    | "alcohol_compliance"
    | "minor_safety"
    | "copyright"
    | "feasibility"
    | "public_display"
    | "trade_eligibility"
    | "medical_claim"
    | "false_promise"
    | "external_transaction";
  risk_level: "low" | "medium" | "high";
  review_status: ReviewStatus;
  reviewer_note?: string;
  created_at: string;
}

export interface SocialAccount {
  id: string;
  platform: string;
  official_url: string;
  icon: string;
  enabled: boolean;
  display_order: number;
  tracking_params?: string;
  backup_url?: string;
  created_at: string;
  updated_at: string;
}

export interface WechatMenuItem {
  label: string;
  children: { label: string; target: string }[];
}

export interface WechatConfig {
  id: string;
  official_account_name: string;
  qr_code_url?: string;
  app_id_set: boolean;
  menu_config: WechatMenuItem[];
  auto_reply_config: { trigger: string; reply: string }[];
  customer_service_url?: string;
  enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface AppConfig {
  id: string;
  ios_download_url?: string;
  android_download_url?: string;
  apk_download_url?: string;
  pwa_enabled: boolean;
  latest_version: string;
  force_update_version?: string;
  changelog: { version: string; date: string; notes: string[] }[];
  show_download_banner: boolean;
  install_prompt_enabled: boolean;
  downloads_enabled: boolean;
  created_at: string;
  updated_at: string;
}

export interface ContentCalendarItem {
  id: string;
  platform: string;
  title: string;
  content: string;
  media_url?: string;
  video_url?: string;
  scheduled_at: string;
  status: "draft" | "scheduled" | "published";
  owner: string;
  related_url?: string;
  related_project_id?: string;
  created_at: string;
  updated_at: string;
}

export type OrderType =
  | "membership"
  | "label_export"
  | "physical_casting"
  | "co_creation"
  | "premium_deposit"
  | "enterprise_project"
  | "design_authorization"
  | "reserve_replenishment"
  | "app_benefit";

export interface Order {
  id: string;
  user_id: string;
  order_type: OrderType;
  title: string;
  amount: number;
  currency: "CNY" | "USD";
  payment_method:
    | "wechat_pay"
    | "alipay"
    | "stripe"
    | "paypal"
    | "offline_transfer"
    | "manual_quote";
  status: "created" | "test_mode" | "awaiting_concierge" | "paid" | "refunded" | "cancelled";
  reference?: string;
  created_at: string;
  updated_at: string;
}

export interface ConciergeLead {
  id: string;
  user_id?: string;
  name: string;
  organization?: string;
  contact: string;
  channel: "maison" | "trade" | "wechat" | "co_create" | "membership";
  scenario: string;
  budget: string;
  status: "new" | "contacted" | "quoting" | "won" | "lost";
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface CaseStudy {
  id: string;
  slug: string;
  title: string;
  title_zh: string;
  category: string;
  client_type: string;
  summary: string;
  story: string[];
  outcome: string;
  emotion_tags: string[];
  featured: boolean;
  created_at: string;
}

export interface BlogPost {
  id: string;
  slug: string;
  title: string;
  title_zh: string;
  excerpt: string;
  body: string[];
  category: string;
  author: string;
  published_at: string;
  featured: boolean;
}

export interface CmsBlock {
  id: string;
  key: string;
  title: string;
  content: string;
  page: string;
  enabled: boolean;
  updated_at: string;
}

export interface LegalDoc {
  id: string;
  slug: string;
  title: string;
  version: string;
  effective_date: string;
  updated_at: string;
}

export interface PlatformSettings {
  id: string;
  site_name: string;
  brand_line_en: string;
  brand_line_zh: string;
  guest_daily_chat: number;
  free_daily_chat: number;
  lite_daily_chat: number;
  pro_daily_chat: number;
  lite_monthly_proposals: number;
  pro_monthly_proposals: number;
  lite_price_month: number;
  lite_price_quarter: number;
  pro_price_month: number;
  pro_price_quarter: number;
  co_create_public_threshold: number;
  co_create_review_threshold: number;
  co_create_label_threshold: number;
  co_create_flavor_threshold: number;
  co_create_enterprise_threshold: number;
  co_create_supply_threshold: number;
  co_create_partner_threshold: number;
  age_gate_enabled: boolean;
  updated_at: string;
}

// ---------------------------------------------------------------------------
// AI structured output
// ---------------------------------------------------------------------------

export type ConciergeMode =
  | "daily"
  | "gift"
  | "spirit"
  | "fragrance"
  | "copy"
  | "style"
  | "recipient"
  | "co_create"
  | "enterprise";

export interface ConciergeInput {
  mode: ConciergeMode;
  message: string;
  emotion?: string;
  recipient?: string;
  scenario?: string;
  budget?: string;
  style?: string;
  locale?: Locale;
}

export type NextActionKey =
  | "save_inspiration"
  | "emotional_card"
  | "label_copy"
  | "add_preferences"
  | "gift_draft"
  | "share"
  | "co_create"
  | "physical_casting"
  | "human_concierge";

export interface ConceptProposal {
  kind: "daily" | "concept";
  emotional_signal: string;
  keywords: string[];
  suggestion?: string;
  liquid_direction?: string;
  scent_direction?: string;
  bottle_direction?: string;
  names?: string[];
  label_copy?: string;
  digital_mark?: string;
  next_actions: NextActionKey[];
}

export interface AiResult {
  reply: string;
  proposal: ConceptProposal;
  model: string;
  tokens_used: number;
  quota_remaining: number | null;
  fallback: boolean;
}
