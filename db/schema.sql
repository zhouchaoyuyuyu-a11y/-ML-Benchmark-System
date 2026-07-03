-- ============================================================================
-- ZOTAIX — Postgres schema (Supabase-compatible)
-- ----------------------------------------------------------------------------
-- Mirrors src/lib/types.ts exactly. The runtime (src/lib/store.ts) uses a
-- seeded in-memory store when DATABASE_URL is unset; when you point
-- DATABASE_URL at a Postgres/Supabase instance, apply this file first and
-- then db/seed.sql.
--
-- Conventions
--   * Primary keys are TEXT, matching the seeded id style ("usr_member",
--     "dft_001", "rsv_001", ...). The app generates ids via newId(prefix).
--   * All timestamps are timestamptz (ISO-8601 strings in the app layer).
--   * Enumerations are enforced with CHECK constraints (kept in lockstep with
--     the union types in src/lib/types.ts).
--   * Structured fields (menu_config, auto_reply_config, design_payload,
--     changelog, story, body, structured proposals) are JSONB.
--   * Tag lists (emotion_tags, favorite_colors, ...) are TEXT[].
--   * Row Level Security statements are included at the bottom, commented,
--     with example policies for user-owned rows (see notes there).
-- ============================================================================

-- ----------------------------------------------------------------------------
-- users
-- ----------------------------------------------------------------------------
create table if not exists users (
  id               text primary key,
  phone            text,
  email            text not null unique,
  password_hash    text,
  nickname         text not null,
  avatar           text,
  user_type        text not null default 'registered'
                     check (user_type in ('guest','registered','member','enterprise','admin')),
  membership_level text not null default 'free'
                     check (membership_level in ('free','lite','pro','enterprise')),
  daily_quota      integer not null default 3,
  used_quota       integer not null default 0,
  created_at       timestamptz not null default now(),
  updated_at       timestamptz not null default now()
);

create index if not exists idx_users_created_at on users (created_at);

-- ----------------------------------------------------------------------------
-- user_profiles — self-expression profile ("brush, not a box")
-- ----------------------------------------------------------------------------
create table if not exists user_profiles (
  id                  text primary key,
  user_id             text not null references users (id) on delete cascade,
  mbti                text,
  zodiac              text,
  blood_type          text,
  age_range           text,
  nickname            text,
  address_style       text,
  favorite_colors     text[],
  scent_preferences   text[],
  alcohol_preferences text[],
  alcohol_tolerance   text,
  non_alcohol_ok      boolean,
  music               text,
  movies              text,
  cities              text,
  literary_imagery    text,
  visual_preferences  text[],
  gift_preferences    text[],
  budget_range        text,
  emotional_state     text,
  common_scenarios    text[],
  personality_tags    text[],
  privacy_level       text not null default 'private'
                        check (privacy_level in ('private','co-create','public')),
  memory_enabled      boolean not null default true,
  created_at          timestamptz not null default now(),
  updated_at          timestamptz not null default now()
);

create index if not exists idx_user_profiles_user_id on user_profiles (user_id);

-- ----------------------------------------------------------------------------
-- relationship_profiles — the people a user customizes for
-- ----------------------------------------------------------------------------
create table if not exists relationship_profiles (
  id              text primary key,
  user_id         text not null references users (id) on delete cascade,
  relation_type   text not null,
  nickname        text not null,
  age_range       text,
  preferences     text,
  important_dates text,
  notes           text,
  privacy_level   text not null default 'private'
                    check (privacy_level in ('private','public')),
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

create index if not exists idx_relationship_profiles_user_id on relationship_profiles (user_id);

-- ----------------------------------------------------------------------------
-- conversations — AI concierge sessions (guests carry visitor_id only)
-- ----------------------------------------------------------------------------
create table if not exists conversations (
  id                text primary key,
  user_id           text references users (id) on delete set null,
  visitor_id        text,
  mode              text not null default 'guest'
                      check (mode in ('guest','registered','member')),
  conversation_type text not null default 'daily'
                      check (conversation_type in ('daily','gift','product','fragrance','wine','enterprise')),
  summary           text not null default '',
  token_usage       integer not null default 0,
  created_at        timestamptz not null default now(),
  updated_at        timestamptz not null default now()
);

create index if not exists idx_conversations_user_id    on conversations (user_id);
create index if not exists idx_conversations_created_at on conversations (created_at);

-- ----------------------------------------------------------------------------
-- messages — turns within a conversation; assistant turns may carry a
-- structured ConceptProposal in `structured`
-- ----------------------------------------------------------------------------
create table if not exists messages (
  id              text primary key,
  conversation_id text not null references conversations (id) on delete cascade,
  role            text not null check (role in ('user','assistant','system')),
  content         text not null,
  structured      jsonb,
  token_usage     integer not null default 0,
  created_at      timestamptz not null default now()
);

create index if not exists idx_messages_conversation_id on messages (conversation_id);
create index if not exists idx_messages_created_at      on messages (created_at);

-- ----------------------------------------------------------------------------
-- ai_usage_logs — quota + cost accounting per generation
-- ----------------------------------------------------------------------------
create table if not exists ai_usage_logs (
  id             text primary key,
  user_id        text references users (id) on delete set null,
  visitor_id     text,
  model          text not null,
  action_type    text not null
                   check (action_type in ('chat','proposal','creative','image','export')),
  tokens_used    integer not null default 0,
  cost_estimate  numeric(10,4) not null default 0,
  quota_consumed integer not null default 0,
  created_at     timestamptz not null default now()
);

create index if not exists idx_ai_usage_logs_user_id    on ai_usage_logs (user_id);
create index if not exists idx_ai_usage_logs_created_at on ai_usage_logs (created_at);

-- ----------------------------------------------------------------------------
-- object_drafts — the personalized objects users create and save FIRST
-- ----------------------------------------------------------------------------
create table if not exists object_drafts (
  id               text primary key,
  user_id          text not null references users (id) on delete cascade,
  object_type      text not null
                     check (object_type in ('spirit','fragrance','bottle','giftbox','label','enterprise_gift')),
  title            text not null,
  scene            text,
  recipient        text,
  budget           text,
  emotion_tags     text[] not null default '{}',
  liquid_direction text,
  scent_direction  text,
  label_copy       text,
  visual_style     text,
  names            text[],
  status           text not null default 'draft'
                     check (status in ('draft','saved','ordered','reviewed')),
  public_visible   boolean not null default false,
  created_at       timestamptz not null default now(),
  updated_at       timestamptz not null default now()
);

create index if not exists idx_object_drafts_user_id    on object_drafts (user_id);
create index if not exists idx_object_drafts_created_at on object_drafts (created_at);
create index if not exists idx_object_drafts_public     on object_drafts (public_visible) where public_visible;

-- ----------------------------------------------------------------------------
-- design_versions — iterated bottle/label/packaging/liquid payloads per draft
-- ----------------------------------------------------------------------------
create table if not exists design_versions (
  id              text primary key,
  object_draft_id text not null references object_drafts (id) on delete cascade,
  version_name    text not null,
  design_payload  jsonb not null default '{}'::jsonb,
  image_url       text,
  model_url       text,
  version_hash    text not null,
  created_at      timestamptz not null default now()
);

create index if not exists idx_design_versions_draft_id on design_versions (object_draft_id);

-- ----------------------------------------------------------------------------
-- memberships — Core Sequence plans and entitlement limits
-- ----------------------------------------------------------------------------
create table if not exists memberships (
  id                       text primary key,
  user_id                  text not null references users (id) on delete cascade,
  plan                     text not null
                             check (plan in ('free','lite','pro','enterprise')),
  monthly_quota            integer not null default 0,
  daily_chat_limit         integer not null default 0,
  premium_generation_limit integer not null default 0,
  image_generation_limit   integer not null default 0,
  export_enabled           boolean not null default false,
  reserve_enabled          boolean not null default false,
  concierge_enabled        boolean not null default false,
  started_at               timestamptz not null default now(),
  expires_at               timestamptz not null
);

create index if not exists idx_memberships_user_id on memberships (user_id);

-- ----------------------------------------------------------------------------
-- co_creation_projects — the co-creation pool
-- ----------------------------------------------------------------------------
create table if not exists co_creation_projects (
  id               text primary key,
  creator_user_id  text not null references users (id) on delete cascade,
  title            text not null,
  concept          text not null,
  product_type     text not null
                     check (product_type in ('wine','fragrance','bottle','giftbox')),
  target_quantity  integer not null default 0,
  current_quantity integer not null default 0,
  supporters       integer not null default 0,
  status           text not null default 'gathering'
                     check (status in ('gathering','review','production','delivered','closed')),
  founder_benefit  text not null default '',
  public_visible   boolean not null default false,
  review_status    text not null default 'pending'
                     check (review_status in ('pending','approved','rejected','revision','escalated')),
  emotion_tags     text[] not null default '{}',
  votes            integer not null default 0,
  created_at       timestamptz not null default now(),
  updated_at       timestamptz not null default now()
);

create index if not exists idx_co_creation_projects_creator on co_creation_projects (creator_user_id);
create index if not exists idx_co_creation_projects_review  on co_creation_projects (review_status, public_visible);
create index if not exists idx_co_creation_projects_created on co_creation_projects (created_at);

-- ----------------------------------------------------------------------------
-- co_creation_members — founders and participants per project
-- ----------------------------------------------------------------------------
create table if not exists co_creation_members (
  id             text primary key,
  project_id     text not null references co_creation_projects (id) on delete cascade,
  user_id        text not null references users (id) on delete cascade,
  role           text not null default 'participant'
                   check (role in ('founder','participant')),
  quantity       integer not null default 1,
  payment_status text not null default 'unpaid'
                   check (payment_status in ('unpaid','reserved','paid','refunded')),
  joined_at      timestamptz not null default now()
);

create index if not exists idx_co_creation_members_project on co_creation_members (project_id);
create index if not exists idx_co_creation_members_user    on co_creation_members (user_id);

-- ----------------------------------------------------------------------------
-- reserve_records — the archive: every kept object gets a ZOTAIX ID,
-- QR/NFC binding, certificate page, aftercare and replenishment status
-- ----------------------------------------------------------------------------
create table if not exists reserve_records (
  id                  text primary key,
  user_id             text not null references users (id) on delete cascade,
  object_draft_id     text references object_drafts (id) on delete set null,
  zotaix_id           text not null unique,
  object_type         text not null
                        check (object_type in ('spirit','fragrance','bottle','giftbox','label','enterprise_gift','badge','co_creation','design_version')),
  object_name         text not null,
  design_version_id   text references design_versions (id) on delete set null,
  emotion_tags        text[] not null default '{}',
  relationship_scene  text,
  product_direction   text,
  label_copy          text,
  scent_direction     text,
  liquid_direction    text,
  visual_style        text,
  batch_id            text,
  qr_nfc_id           text not null,
  certificate_url     text,
  privacy_level       text not null default 'private'
                        check (privacy_level in ('private','public')),
  co_create_eligible  boolean not null default false,
  delivery_status     text not null default 'digital'
                        check (delivery_status in ('digital','pending_review','in_production','delivered')),
  repurchase_eligible boolean not null default false,
  aftercare_status    text not null default 'none'
                        check (aftercare_status in ('active','expired','none')),
  created_at          timestamptz not null default now(),
  updated_at          timestamptz not null default now()
);

create index if not exists idx_reserve_records_user_id    on reserve_records (user_id);
create index if not exists idx_reserve_records_created_at on reserve_records (created_at);
create index if not exists idx_reserve_records_qr         on reserve_records (qr_nfc_id);

-- ----------------------------------------------------------------------------
-- trade_requests — quotes, authorizations, enterprise projects,
-- collaborations, replenishments; always human-reviewed before quoting
-- ----------------------------------------------------------------------------
create table if not exists trade_requests (
  id                  text primary key,
  user_id             text not null references users (id) on delete cascade,
  object_draft_id     text references object_drafts (id) on delete set null,
  request_type        text not null
                        check (request_type in ('quote','authorization','enterprise','collaboration','replenishment')),
  organization        text,
  contact             text,
  quantity            integer not null default 0,
  budget              text not null default '',
  deadline            text,
  delivery_region     text,
  liquid_direction    text,
  scent_direction     text,
  bottle_direction    text,
  packaging_direction text,
  sample_path         text,
  invoice_required    boolean not null default false,
  logistics_notes     text,
  compliance_status   text not null default 'unchecked'
                        check (compliance_status in ('unchecked','passed','flagged')),
  human_review_status text not null default 'pending'
                        check (human_review_status in ('pending','approved','rejected','revision','escalated')),
  quote_status        text not null default 'none'
                        check (quote_status in ('none','drafting','sent','accepted','declined')),
  notes               text,
  created_at          timestamptz not null default now(),
  updated_at          timestamptz not null default now()
);

create index if not exists idx_trade_requests_user_id    on trade_requests (user_id);
create index if not exists idx_trade_requests_created_at on trade_requests (created_at);
create index if not exists idx_trade_requests_review     on trade_requests (human_review_status);

-- ----------------------------------------------------------------------------
-- moderation_logs — compliance and safety review trail
-- ----------------------------------------------------------------------------
create table if not exists moderation_logs (
  id            text primary key,
  user_id       text references users (id) on delete set null,
  object_id     text not null,
  content_type  text not null,
  risk_type     text not null
                  check (risk_type in ('sensitive','alcohol_compliance','minor_safety','copyright','feasibility','public_display','trade_eligibility','medical_claim','false_promise','external_transaction')),
  risk_level    text not null check (risk_level in ('low','medium','high')),
  review_status text not null default 'pending'
                  check (review_status in ('pending','approved','rejected','revision','escalated')),
  reviewer_note text,
  created_at    timestamptz not null default now()
);

create index if not exists idx_moderation_logs_object     on moderation_logs (object_id);
create index if not exists idx_moderation_logs_review     on moderation_logs (review_status);
create index if not exists idx_moderation_logs_created_at on moderation_logs (created_at);

-- ----------------------------------------------------------------------------
-- social_accounts — the global social matrix shown on /social
-- ----------------------------------------------------------------------------
create table if not exists social_accounts (
  id              text primary key,
  platform        text not null,
  official_url    text not null,
  icon            text not null,
  enabled         boolean not null default true,
  display_order   integer not null default 0,
  tracking_params text,
  backup_url      text,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

create index if not exists idx_social_accounts_order on social_accounts (display_order);

-- ----------------------------------------------------------------------------
-- wechat_config — Official Account menu, auto-replies, service entry
-- (single row; admin-editable via /api/admin/config)
-- ----------------------------------------------------------------------------
create table if not exists wechat_config (
  id                    text primary key,
  official_account_name text not null,
  qr_code_url           text,
  app_id_set            boolean not null default false,
  menu_config           jsonb not null default '[]'::jsonb,
  auto_reply_config     jsonb not null default '[]'::jsonb,
  customer_service_url  text,
  enabled               boolean not null default true,
  created_at            timestamptz not null default now(),
  updated_at            timestamptz not null default now()
);

-- ----------------------------------------------------------------------------
-- app_config — app distribution, versioning, changelog (single row)
-- ----------------------------------------------------------------------------
create table if not exists app_config (
  id                    text primary key,
  ios_download_url      text,
  android_download_url  text,
  apk_download_url      text,
  pwa_enabled           boolean not null default true,
  latest_version        text not null default '1.0.0',
  force_update_version  text,
  changelog             jsonb not null default '[]'::jsonb,
  show_download_banner  boolean not null default true,
  install_prompt_enabled boolean not null default true,
  downloads_enabled     boolean not null default true,
  created_at            timestamptz not null default now(),
  updated_at            timestamptz not null default now()
);

-- ----------------------------------------------------------------------------
-- content_calendar — planned/published posts across platforms
-- ----------------------------------------------------------------------------
create table if not exists content_calendar (
  id                 text primary key,
  platform           text not null,
  title              text not null,
  content            text not null,
  media_url          text,
  video_url          text,
  scheduled_at       timestamptz not null,
  status             text not null default 'draft'
                       check (status in ('draft','scheduled','published')),
  owner              text not null default '',
  related_url        text,
  related_project_id text,
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now()
);

create index if not exists idx_content_calendar_scheduled on content_calendar (scheduled_at);
create index if not exists idx_content_calendar_status    on content_calendar (status);

-- ----------------------------------------------------------------------------
-- orders — memberships, exports, castings, deposits; payment methods degrade
-- to concierge confirmation when a gateway is not configured
-- ----------------------------------------------------------------------------
create table if not exists orders (
  id             text primary key,
  user_id        text not null references users (id) on delete cascade,
  order_type     text not null
                   check (order_type in ('membership','label_export','physical_casting','co_creation','premium_deposit','enterprise_project','design_authorization','reserve_replenishment','app_benefit')),
  title          text not null,
  amount         numeric(12,2) not null default 0,
  currency       text not null default 'CNY' check (currency in ('CNY','USD')),
  payment_method text not null
                   check (payment_method in ('wechat_pay','alipay','stripe','paypal','offline_transfer','manual_quote')),
  status         text not null default 'created'
                   check (status in ('created','test_mode','awaiting_concierge','paid','refunded','cancelled')),
  reference      text,
  created_at     timestamptz not null default now(),
  updated_at     timestamptz not null default now()
);

create index if not exists idx_orders_user_id    on orders (user_id);
create index if not exists idx_orders_created_at on orders (created_at);

-- ----------------------------------------------------------------------------
-- concierge_leads — human-concierge pipeline (maison / trade / wechat / ...)
-- ----------------------------------------------------------------------------
create table if not exists concierge_leads (
  id           text primary key,
  user_id      text references users (id) on delete set null,
  name         text not null,
  organization text,
  contact      text not null,
  channel      text not null
                 check (channel in ('maison','trade','wechat','co_create','membership')),
  scenario     text not null default '',
  budget       text not null default '',
  status       text not null default 'new'
                 check (status in ('new','contacted','quoting','won','lost')),
  notes        text,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

create index if not exists idx_concierge_leads_status     on concierge_leads (status);
create index if not exists idx_concierge_leads_created_at on concierge_leads (created_at);

-- ----------------------------------------------------------------------------
-- case_studies — published delivery stories (story is a JSONB string array)
-- ----------------------------------------------------------------------------
create table if not exists case_studies (
  id           text primary key,
  slug         text not null unique,
  title        text not null,
  title_zh     text not null,
  category     text not null,
  client_type  text not null,
  summary      text not null,
  story        jsonb not null default '[]'::jsonb,
  outcome      text not null,
  emotion_tags text[] not null default '{}',
  featured     boolean not null default false,
  created_at   timestamptz not null default now()
);

create index if not exists idx_case_studies_featured on case_studies (featured);

-- ----------------------------------------------------------------------------
-- blog_posts — editorial (body is a JSONB string array of paragraphs)
-- ----------------------------------------------------------------------------
create table if not exists blog_posts (
  id           text primary key,
  slug         text not null unique,
  title        text not null,
  title_zh     text not null,
  excerpt      text not null,
  body         jsonb not null default '[]'::jsonb,
  category     text not null,
  author       text not null,
  published_at timestamptz not null,
  featured     boolean not null default false
);

create index if not exists idx_blog_posts_published on blog_posts (published_at);

-- ----------------------------------------------------------------------------
-- cms_blocks — admin-editable copy blocks keyed per page
-- ----------------------------------------------------------------------------
create table if not exists cms_blocks (
  id         text primary key,
  key        text not null unique,
  title      text not null,
  content    text not null,
  page       text not null,
  enabled    boolean not null default true,
  updated_at timestamptz not null default now()
);

create index if not exists idx_cms_blocks_page on cms_blocks (page);

-- ----------------------------------------------------------------------------
-- legal_docs — registry of legal documents rendered under /legal/[slug]
-- ----------------------------------------------------------------------------
create table if not exists legal_docs (
  id             text primary key,
  slug           text not null unique,
  title          text not null,
  version        text not null,
  effective_date text not null,
  updated_at     timestamptz not null default now()
);

-- ----------------------------------------------------------------------------
-- platform_settings — quotas, pricing, co-creation thresholds (single row)
-- ----------------------------------------------------------------------------
create table if not exists platform_settings (
  id                            text primary key,
  site_name                     text not null default 'ZOTAIX',
  brand_line_en                 text not null default '',
  brand_line_zh                 text not null default '',
  guest_daily_chat              integer not null default 3,
  free_daily_chat               integer not null default 3,
  lite_daily_chat               integer not null default 15,
  pro_daily_chat                integer not null default 50,
  lite_monthly_proposals        integer not null default 20,
  pro_monthly_proposals         integer not null default 80,
  lite_price_month              integer not null default 19,
  lite_price_quarter            integer not null default 49,
  pro_price_month               integer not null default 49,
  pro_price_quarter             integer not null default 128,
  co_create_public_threshold    integer not null default 10,
  co_create_review_threshold    integer not null default 30,
  co_create_label_threshold     integer not null default 50,
  co_create_flavor_threshold    integer not null default 100,
  co_create_enterprise_threshold integer not null default 300,
  co_create_supply_threshold    integer not null default 500,
  co_create_partner_threshold   integer not null default 1000,
  age_gate_enabled              boolean not null default true,
  updated_at                    timestamptz not null default now()
);

-- ============================================================================
-- Row Level Security (Supabase)
-- ----------------------------------------------------------------------------
-- The Next.js server talks to Postgres with the service role (bypasses RLS),
-- so the platform works with RLS on or off. Enable RLS before exposing any
-- table to Supabase client-side keys.
--
-- NOTE ON IDENTITY MAPPING: primary keys here are app-generated TEXT ids
-- ("usr_..."), not auth.uid() UUIDs. If you adopt Supabase Auth, either
--   (a) store auth.uid()::text as users.id when creating users, or
--   (b) add a users.auth_id uuid column and join through it in policies.
-- The example policies below assume option (a): auth.uid()::text = user_id.
--
-- Enable per table (uncomment to activate):
-- alter table users                 enable row level security;
-- alter table user_profiles         enable row level security;
-- alter table relationship_profiles enable row level security;
-- alter table conversations         enable row level security;
-- alter table messages              enable row level security;
-- alter table ai_usage_logs         enable row level security;
-- alter table object_drafts         enable row level security;
-- alter table design_versions       enable row level security;
-- alter table memberships           enable row level security;
-- alter table co_creation_projects  enable row level security;
-- alter table co_creation_members   enable row level security;
-- alter table reserve_records       enable row level security;
-- alter table trade_requests        enable row level security;
-- alter table moderation_logs       enable row level security;
-- alter table social_accounts       enable row level security;
-- alter table wechat_config         enable row level security;
-- alter table app_config            enable row level security;
-- alter table content_calendar      enable row level security;
-- alter table orders                enable row level security;
-- alter table concierge_leads       enable row level security;
-- alter table case_studies          enable row level security;
-- alter table blog_posts            enable row level security;
-- alter table cms_blocks            enable row level security;
-- alter table legal_docs            enable row level security;
-- alter table platform_settings     enable row level security;
--
-- Example: user-owned rows — owner can read/write their own records.
-- create policy "own_profile_rw" on user_profiles
--   for all
--   using (auth.uid()::text = user_id)
--   with check (auth.uid()::text = user_id);
--
-- create policy "own_drafts_rw" on object_drafts
--   for all
--   using (auth.uid()::text = user_id)
--   with check (auth.uid()::text = user_id);
--
-- Example: public read where the row is explicitly public.
-- create policy "drafts_public_read" on object_drafts
--   for select
--   using (public_visible = true);
--
-- create policy "reserve_public_read" on reserve_records
--   for select
--   using (privacy_level = 'public');
--
-- create policy "reserve_owner_rw" on reserve_records
--   for all
--   using (auth.uid()::text = user_id)
--   with check (auth.uid()::text = user_id);
--
-- Example: published content is world-readable; writes stay service-role only.
-- create policy "cases_read"  on case_studies for select using (true);
-- create policy "posts_read"  on blog_posts   for select using (true);
-- create policy "cms_read"    on cms_blocks   for select using (enabled = true);
-- create policy "legal_read"  on legal_docs   for select using (true);
-- create policy "social_read" on social_accounts for select using (enabled = true);
--
-- Example: co-creation pool — approved public projects readable by anyone,
-- creators manage their own rows.
-- create policy "projects_public_read" on co_creation_projects
--   for select
--   using (public_visible = true and review_status = 'approved');
-- create policy "projects_owner_rw" on co_creation_projects
--   for all
--   using (auth.uid()::text = creator_user_id)
--   with check (auth.uid()::text = creator_user_id);
-- ============================================================================
