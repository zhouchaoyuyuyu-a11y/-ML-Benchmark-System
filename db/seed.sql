-- ============================================================================
-- ZOTAIX — seed data (mirrors src/lib/seed.ts exactly)
-- ----------------------------------------------------------------------------
-- Apply after db/schema.sql. Idempotent: every insert uses
-- ON CONFLICT DO NOTHING, so re-running is safe.
-- Timestamps reused from seed.ts:
--   T  = 2026-06-01T10:00:00.000Z
--   T2 = 2026-06-18T10:00:00.000Z
--   T3 = 2026-06-29T10:00:00.000Z
-- ============================================================================

-- ----------------------------------------------------------------------------
-- users (5)
-- ----------------------------------------------------------------------------
insert into users (id, email, nickname, user_type, membership_level, daily_quota, used_quota, created_at, updated_at) values
  ('usr_admin',  'admin@zotaix.demo',                    'ZOTAIX Operations',     'admin',      'pro',        999, 0,  '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z'),
  ('usr_member', 'member@zotaix.demo',                   'Sequence Member Lin',   'member',     'pro',        50,  6,  '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z'),
  ('usr_lite',   'lite@zotaix.demo',                     'Aria W.',               'member',     'lite',       15,  3,  '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z'),
  ('usr_free',   'user@zotaix.demo',                     'Night Owl Chen',        'registered', 'free',       3,   1,  '2026-06-18T10:00:00.000Z', '2026-06-29T10:00:00.000Z'),
  ('usr_ent',    'gifting@meridian-hotels.example',      'Meridian Hotels Group', 'enterprise', 'enterprise', 200, 12, '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z')
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- user_profiles (2)
-- ----------------------------------------------------------------------------
insert into user_profiles (
  id, user_id, mbti, zodiac, blood_type, age_range, nickname, address_style,
  favorite_colors, scent_preferences, alcohol_preferences, alcohol_tolerance,
  non_alcohol_ok, music, movies, cities, literary_imagery, visual_preferences,
  gift_preferences, budget_range, emotional_state, common_scenarios,
  personality_tags, privacy_level, memory_enabled, created_at, updated_at
) values
  (
    'prf_member', 'usr_member', 'INFJ', 'Scorpio', 'O', '25-34', 'Lin',
    'First name, no honorifics',
    array['Ink blue','Champagne gold'],
    array['Woody','Cold incense','White tea'],
    array['Single malt','Umeshu','Low-ABV botanical'],
    'Moderate', true,
    'Post-rock, Ryuichi Sakamoto',
    'In the Mood for Love, Arrival',
    'Kyoto, Chongqing, Reykjavik',
    'Borges'' library, midnight harbors',
    array['Eastern','Restrained','Futuristic'],
    array['Meaning over price','Objects with stories'],
    '300-1500 RMB',
    'Rebuilding after a demanding quarter',
    array['Late-night work','Gifts for close friends'],
    array['Order-seeking','Quiet intensity','Archivist'],
    'co-create', true,
    '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  ),
  (
    'prf_free', 'usr_free', 'ENFP', 'Gemini', null, '18-24', null, null,
    array['Sunset orange'],
    array['Citrus','Sea salt'],
    array['Fruit beer','Sparkling low-ABV'],
    'Light', true,
    null, null, null, null,
    array['Playful','Sweet-cool'],
    null,
    '50-300 RMB',
    'Exam season pressure',
    array['Birthday gifts','Post-exam celebration'],
    array['Spark','Momentum'],
    'private', true,
    '2026-06-18T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  )
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- relationship_profiles (2)
-- ----------------------------------------------------------------------------
insert into relationship_profiles (
  id, user_id, relation_type, nickname, age_range, preferences, important_dates,
  notes, privacy_level, created_at, updated_at
) values
  (
    'rel_001', 'usr_member', 'Close friend', 'A-Yao', '25-34',
    'Whisky, mountain hiking, film photography', 'Birthday 09-21',
    'Just moved to Shanghai; misses Chengdu.', 'private',
    '2026-06-01T10:00:00.000Z', '2026-06-18T10:00:00.000Z'
  ),
  (
    'rel_002', 'usr_member', 'Partner', 'S', '25-34',
    'White florals, minimal design, oolong tea', 'Anniversary 11-02',
    'Prefers understated gifts with a written note.', 'private',
    '2026-06-01T10:00:00.000Z', '2026-06-18T10:00:00.000Z'
  )
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- conversations (3)
-- ----------------------------------------------------------------------------
insert into conversations (id, user_id, mode, conversation_type, summary, token_usage, created_at, updated_at) values
  (
    'cnv_001', 'usr_member', 'member', 'gift',
    'Anniversary gift for partner S — restrained white-floral fragrance direction with engraved bottle.',
    4210, '2026-06-18T10:00:00.000Z', '2026-06-18T10:00:00.000Z'
  ),
  (
    'cnv_002', 'usr_free', 'registered', 'daily',
    'Exam-season stress check-in; suggested ''Night Study Ration'' emotional card.',
    830, '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  ),
  (
    'cnv_003', 'usr_ent', 'member', 'enterprise',
    '300-unit client appreciation gift boxes for hotel VIPs, budget 450 RMB/unit.',
    6120, '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  )
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- messages (2)
-- ----------------------------------------------------------------------------
insert into messages (id, conversation_id, role, content, structured, token_usage, created_at) values
  (
    'msg_001', 'cnv_001', 'user',
    'Our anniversary is coming. S likes white florals and very quiet design. Budget around 1200.',
    null, 42, '2026-06-18T10:00:00.000Z'
  ),
  (
    'msg_002', 'cnv_001', 'assistant',
    'A restrained anniversary object: white floral top, warm woody base, matte porcelain-white bottle with a single engraved line.',
    '{
      "kind": "concept",
      "emotional_signal": "Quiet devotion, eleven years of steady light",
      "keywords": ["white floral", "restrained", "anniversary"],
      "liquid_direction": "Low-ABV white-tea infusion base with plum blossom accents",
      "scent_direction": "Gardenia and white tea over sandalwood",
      "bottle_direction": "Matte porcelain white, tall slender profile, engraved base ring",
      "names": ["White Interval", "Eleventh Light", "S · Quiet Bloom"],
      "label_copy": "Some light does not flare. It stays.",
      "digital_mark": "Anniversary Seal · No. 011",
      "next_actions": ["save_inspiration", "gift_draft", "physical_casting", "human_concierge"]
    }'::jsonb,
    380, '2026-06-18T10:00:00.000Z'
  )
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- ai_usage_logs (4)
-- ----------------------------------------------------------------------------
insert into ai_usage_logs (id, user_id, visitor_id, model, action_type, tokens_used, cost_estimate, quota_consumed, created_at) values
  ('aiu_001', 'usr_member', null,       'atelier-structured', 'proposal', 380,  0.004, 1, '2026-06-18T10:00:00.000Z'),
  ('aiu_002', 'usr_free',   null,       'atelier-lite',       'chat',     120,  0.001, 1, '2026-06-29T10:00:00.000Z'),
  ('aiu_003', 'usr_ent',    null,       'atelier-structured', 'creative', 2200, 0.03,  3, '2026-06-29T10:00:00.000Z'),
  ('aiu_004', null,         'vis_9f2a', 'atelier-lite',       'chat',     90,   0.001, 1, '2026-06-29T10:00:00.000Z')
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- object_drafts (4)
-- ----------------------------------------------------------------------------
insert into object_drafts (
  id, user_id, object_type, title, scene, recipient, budget, emotion_tags,
  liquid_direction, scent_direction, label_copy, visual_style, names, status,
  public_visible, created_at, updated_at
) values
  (
    'dft_001', 'usr_member', 'spirit', 'Midnight Rebuild',
    'Late-night work, end of a hard quarter', 'Self', '600 RMB',
    array['order','recovery','quiet strength'],
    'Smoked plum and oolong over a clean grain spirit, 32% ABV',
    'Cold incense, wet stone, a trace of yuzu',
    'The world can stay chaotic. Tonight, I rebuild my own order.',
    'Ink-black matte glass, champagne-gold serif, vertical typesetting',
    array['Midnight Rebuild','Order 03:00','Quiet Foundry'],
    'saved', true, '2026-06-18T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  ),
  (
    'dft_002', 'usr_member', 'fragrance', 'White Interval',
    'Anniversary', 'Partner S', '1200 RMB',
    array['devotion','calm','white floral'],
    null,
    'Gardenia and white tea over sandalwood',
    'Some light does not flare. It stays.',
    'Matte porcelain white, minimal, engraved base ring',
    array['White Interval','Eleventh Light'],
    'saved', false, '2026-06-18T10:00:00.000Z', '2026-06-18T10:00:00.000Z'
  ),
  (
    'dft_003', 'usr_free', 'label', 'Night Study Ration',
    'Exam season', 'Self', '80 RMB',
    array['persistence','humor','exam'],
    null, null,
    'One more page. Then the whole sky.',
    'Sunset-orange gradient, handwritten accent, sticker energy',
    null,
    'draft', true, '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  ),
  (
    'dft_004', 'usr_ent', 'enterprise_gift', 'Meridian VIP Appreciation Box',
    'Hotel VIP client appreciation, year-end', 'Top-tier guests', '450 RMB / unit × 300',
    array['gratitude','prestige','continuity'],
    'Aged huangjiu blend with citrus-peel finish',
    'Amber, hinoki, warm spice room mist',
    'A year measured in arrivals. Thank you for returning.',
    'Deep bronze and ivory, embossed monogram, magnetic gift box',
    null,
    'reviewed', false, '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  )
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- design_versions (3)
-- ----------------------------------------------------------------------------
insert into design_versions (id, object_draft_id, version_name, design_payload, version_hash, created_at) values
  (
    'ver_001', 'dft_001', 'v1 · Ink Foundation',
    '{
      "bottle": "Tall shoulderless cylinder, matte ink glass",
      "label": "Vertical gold serif on black wrap",
      "packaging": "Slipcase with wax seal",
      "liquid": "Smoked plum + oolong, 32% ABV"
    }'::jsonb,
    'zx-8f31c2a0', '2026-06-18T10:00:00.000Z'
  ),
  (
    'ver_002', 'dft_001', 'v2 · Gold Meridian',
    '{
      "bottle": "Tall cylinder with engraved meridian ring",
      "label": "Horizontal band, deep gold foil",
      "packaging": "Magnetic box, night-blue interior",
      "liquid": "Smoked plum + oolong, 30% ABV, softer finish"
    }'::jsonb,
    'zx-2b77d914', '2026-06-29T10:00:00.000Z'
  ),
  (
    'ver_003', 'dft_004', 'v1 · Bronze Monogram',
    '{
      "bottle": "Wide-shoulder decanter, bronze cap",
      "label": "Ivory wrap, embossed M monogram",
      "packaging": "Two-bottle magnetic case + room mist vial",
      "liquid": "Aged huangjiu blend, citrus-peel finish"
    }'::jsonb,
    'zx-51e0aa37', '2026-06-29T10:00:00.000Z'
  )
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- memberships (2)
-- ----------------------------------------------------------------------------
insert into memberships (
  id, user_id, plan, monthly_quota, daily_chat_limit, premium_generation_limit,
  image_generation_limit, export_enabled, reserve_enabled, concierge_enabled,
  started_at, expires_at
) values
  ('mem_001', 'usr_member', 'pro',  80, 50, 80, 30, true,  true, true,  '2026-06-01T10:00:00.000Z', '2026-09-01T00:00:00.000Z'),
  ('mem_002', 'usr_lite',   'lite', 20, 15, 20, 5,  false, true, false, '2026-06-18T10:00:00.000Z', '2026-07-18T00:00:00.000Z')
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- co_creation_projects (3)
-- ----------------------------------------------------------------------------
insert into co_creation_projects (
  id, creator_user_id, title, concept, product_type, target_quantity,
  current_quantity, supporters, status, founder_benefit, public_visible,
  review_status, emotion_tags, votes, created_at, updated_at
) values
  (
    'ccp_001', 'usr_member',
    'Order 03:00 — a bottle for everyone rebuilding at night',
    'A co-created low-smoke spirit for late-night rebuilders: smoked plum, oolong, and a label that reads like a quiet manifesto. Founder edition carries an engraved meridian ring.',
    'wine', 100, 64, 38, 'gathering',
    'Founder Edition serial + engraved name + exclusive QR archive page',
    true, 'approved', array['night','rebuild','order'], 214,
    '2026-06-18T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  ),
  (
    'ccp_002', 'usr_free',
    'Exam Season Survival Kit — zero-proof sparkling ration',
    'A zero-proof sparkling supply with citrus and sea-salt notes, sticker-style labels, and an emotional card pack for study groups.',
    'giftbox', 50, 41, 41, 'gathering',
    'Founder badge + name printed on the card pack colophon',
    true, 'approved', array['exam','persistence','citrus'], 96,
    '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  ),
  (
    'ccp_003', 'usr_member',
    'City Fog Fragrance — Chongqing edition',
    'A fragrance direction built from Chongqing imagery: river fog, pepper warmth, neon reflection. Proposed as a 30-unit fragrance supply run.',
    'fragrance', 30, 12, 12, 'gathering',
    'Founder digital mark + first-pour reservation',
    true, 'pending', array['city','fog','warmth'], 47,
    '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  )
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- co_creation_members (3)
-- ----------------------------------------------------------------------------
insert into co_creation_members (id, project_id, user_id, role, quantity, payment_status, joined_at) values
  ('ccm_001', 'ccp_001', 'usr_member', 'founder',     2, 'reserved', '2026-06-18T10:00:00.000Z'),
  ('ccm_002', 'ccp_001', 'usr_free',   'participant', 1, 'reserved', '2026-06-29T10:00:00.000Z'),
  ('ccm_003', 'ccp_002', 'usr_free',   'founder',     3, 'reserved', '2026-06-29T10:00:00.000Z')
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- reserve_records (3)
-- ----------------------------------------------------------------------------
insert into reserve_records (
  id, user_id, object_draft_id, zotaix_id, object_type, object_name,
  design_version_id, emotion_tags, relationship_scene, product_direction,
  label_copy, scent_direction, liquid_direction, batch_id, qr_nfc_id,
  certificate_url, privacy_level, co_create_eligible, delivery_status,
  repurchase_eligible, aftercare_status, created_at, updated_at
) values
  (
    'rsv_001', 'usr_member', 'dft_001', 'ZX-2026-0611-0001', 'spirit', 'Midnight Rebuild',
    'ver_002', array['order','recovery'], 'Self · end of quarter',
    'Small-batch smoked plum spirit on standard base liquid',
    'The world can stay chaotic. Tonight, I rebuild my own order.',
    'Cold incense, wet stone, yuzu trace',
    'Smoked plum + oolong, 30% ABV',
    'BATCH-M11', 'QR-ZX-8F31C2A0', '/reserve/rsv_001',
    'public', true, 'in_production', true, 'active',
    '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  ),
  (
    'rsv_002', 'usr_member', 'dft_002', 'ZX-2026-0618-0002', 'fragrance', 'White Interval',
    null, array['devotion','calm'], 'Partner · anniversary',
    'Fragrance direction archived for atelier composition',
    'Some light does not flare. It stays.',
    'Gardenia and white tea over sandalwood',
    null, null, 'QR-ZX-4A19E3B7', null,
    'private', false, 'digital', true, 'none',
    '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  ),
  (
    'rsv_003', 'usr_free', null, 'ZX-2026-0629-0003', 'badge', 'Night Study Ration · Digital Mark',
    null, array['persistence','exam'], 'Self · exam season',
    null,
    'One more page. Then the whole sky.',
    null, null, null, 'QR-ZX-C2D40F11', null,
    'public', true, 'digital', false, 'none',
    '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  )
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- trade_requests (2)
-- ----------------------------------------------------------------------------
insert into trade_requests (
  id, user_id, object_draft_id, request_type, organization, contact, quantity,
  budget, deadline, delivery_region, liquid_direction, scent_direction,
  bottle_direction, packaging_direction, sample_path, invoice_required,
  logistics_notes, compliance_status, human_review_status, quote_status, notes,
  created_at, updated_at
) values
  (
    'trq_001', 'usr_ent', 'dft_004', 'enterprise', 'Meridian Hotels Group',
    'gifting@meridian-hotels.example', 300, '135,000 RMB total', '2026-12-01',
    'Shanghai / Beijing / Chengdu',
    'Aged huangjiu blend, citrus-peel finish',
    'Amber, hinoki, warm spice room mist',
    'Wide-shoulder decanter, bronze cap',
    'Two-bottle magnetic case + mist vial',
    '3 pre-production samples by 2026-09-15', true,
    'Staggered delivery to three cities, temperature-controlled',
    'passed', 'approved', 'sent',
    'Concierge: Wen. Quote ZX-Q-2026-118 sent 06-30.',
    '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  ),
  (
    'trq_002', 'usr_member', 'dft_001', 'quote', null, null, 12,
    '7,200 RMB', '2026-08-20', 'Hangzhou',
    null, null, null, null, null, false, null,
    'unchecked', 'pending', 'drafting', null,
    '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  )
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- moderation_logs (3)
-- ----------------------------------------------------------------------------
insert into moderation_logs (id, user_id, object_id, content_type, risk_type, risk_level, review_status, reviewer_note, created_at) values
  (
    'mod_001', 'usr_free', 'ccp_002', 'co_creation_project', 'minor_safety', 'low', 'approved',
    'Zero-proof product; exam framing acceptable, no alcohol imagery near minors.',
    '2026-06-29T10:00:00.000Z'
  ),
  (
    'mod_002', 'usr_member', 'ccp_003', 'co_creation_project', 'feasibility', 'medium', 'pending',
    'Awaiting atelier confirmation on pepper-note stability.',
    '2026-06-29T10:00:00.000Z'
  ),
  (
    'mod_003', null, 'dft_003', 'object_draft', 'public_display', 'low', 'approved', null,
    '2026-06-29T10:00:00.000Z'
  )
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- social_accounts (8)
-- ----------------------------------------------------------------------------
insert into social_accounts (id, platform, official_url, icon, enabled, display_order, tracking_params, created_at, updated_at) values
  ('soc_ig',  'Instagram',   'https://instagram.com/zotaix.official',   'instagram', true, 1, 'utm_source=instagram', '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z'),
  ('soc_tt',  'TikTok',      'https://tiktok.com/@zotaix',              'tiktok',    true, 2, 'utm_source=tiktok',    '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z'),
  ('soc_x',   'X / Twitter', 'https://x.com/zotaix',                    'x',         true, 3, 'utm_source=x',         '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z'),
  ('soc_yt',  'YouTube',     'https://youtube.com/@zotaix',             'youtube',   true, 4, 'utm_source=youtube',   '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z'),
  ('soc_li',  'LinkedIn',    'https://linkedin.com/company/zotaix',     'linkedin',  true, 5, 'utm_source=linkedin',  '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z'),
  ('soc_fb',  'Facebook',    'https://facebook.com/zotaix',             'facebook',  true, 6, 'utm_source=facebook',  '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z'),
  ('soc_pin', 'Pinterest',   'https://pinterest.com/zotaix',            'pinterest', true, 7, 'utm_source=pinterest', '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z'),
  ('soc_th',  'Threads',     'https://threads.net/@zotaix.official',    'threads',   true, 8, 'utm_source=threads',   '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z')
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- wechat_config (1)
-- ----------------------------------------------------------------------------
insert into wechat_config (
  id, official_account_name, qr_code_url, app_id_set, menu_config,
  auto_reply_config, customer_service_url, enabled, created_at, updated_at
) values (
  'wx_001', 'ZOTAIX 卓序', '', false,
  '[
    {
      "label": "AI Concierge",
      "children": [
        {"label": "Today''s Emotion", "target": "/concierge?mode=daily"},
        {"label": "Gift Inspiration", "target": "/concierge?mode=gift"},
        {"label": "Fragrance Test", "target": "/concierge?mode=fragrance"},
        {"label": "Spirit Test", "target": "/concierge?mode=spirit"}
      ]
    },
    {
      "label": "Co-Creation & Membership",
      "children": [
        {"label": "Co-Creation Pool", "target": "/co-create"},
        {"label": "My Reserve", "target": "/reserve"},
        {"label": "Core Sequence", "target": "/membership"},
        {"label": "Digital Marks", "target": "/profile"}
      ]
    },
    {
      "label": "Premium Customization",
      "children": [
        {"label": "Enterprise Gifting", "target": "/maison"},
        {"label": "Brand Collaboration", "target": "/maison#collaboration"},
        {"label": "Human Concierge", "target": "/maison#concierge"},
        {"label": "Contact Us", "target": "/legal/contact"}
      ]
    }
  ]'::jsonb,
  '[
    {"trigger": "__follow__", "reply": "Welcome to ZOTAIX 卓序. Reply 情绪 for today''s emotional check-in, 礼物 for gift inspiration, 共创 for the co-creation pool, 高定 for premium customization, or APP for the app download."},
    {"trigger": "情绪", "reply": "Tell me one sentence about how today feels, and the AI concierge will answer with keywords and a light suggestion: "},
    {"trigger": "礼物", "reply": "Who is the gift for, what is the moment, and what is the budget? The concierge will propose directions: "},
    {"trigger": "共创", "reply": "The co-creation pool is gathering projects — 10 people open a public page; 100 bottles unlock flavor review: "},
    {"trigger": "高定", "reply": "Premium and enterprise gifting go through a human concierge. Leave your scenario and budget, and a concierge will reply within one business day."},
    {"trigger": "APP", "reply": "Download the ZOTAIX app or install the web app: "},
    {"trigger": "客服", "reply": "A human concierge is available on business days 10:00–19:00 (CST). You can also write to concierge@zotaix.example."}
  ]'::jsonb,
  '', true, '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
)
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- app_config (1)
-- ----------------------------------------------------------------------------
insert into app_config (
  id, ios_download_url, android_download_url, apk_download_url, pwa_enabled,
  latest_version, force_update_version, changelog, show_download_banner,
  install_prompt_enabled, downloads_enabled, created_at, updated_at
) values (
  'app_001', '', '', '', true, '1.4.0', '1.2.0',
  '[
    {"version": "1.4.0", "date": "2026-06-20", "notes": ["Reserve certificates with QR binding", "Co-creation pool voting", "WeChat sharing cards"]},
    {"version": "1.3.0", "date": "2026-05-12", "notes": ["Studio bottle preview", "Membership quota center"]},
    {"version": "1.2.0", "date": "2026-04-02", "notes": ["AI concierge structured proposals", "Emotional cards"]}
  ]'::jsonb,
  true, true, true, '2026-06-01T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
)
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- content_calendar (4)
-- ----------------------------------------------------------------------------
insert into content_calendar (
  id, platform, title, content, scheduled_at, status, owner, related_url,
  related_project_id, created_at, updated_at
) values
  (
    'cal_001', 'Instagram', 'Reserve certificate walkthrough',
    'Every ZOTAIX object gets an identity. 60-second walkthrough of a Reserve certificate, from emotion to QR.',
    '2026-07-08T12:00:00.000Z', 'scheduled', 'Brand · Mia', '/reserve', null,
    '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  ),
  (
    'cal_002', 'TikTok', 'POV: your breakup gets a label',
    'Emotional supply line — from one sentence to a shareable label. Duet-friendly format.',
    '2026-07-10T09:00:00.000Z', 'draft', 'Brand · Ken', '/supply', null,
    '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  ),
  (
    'cal_003', 'LinkedIn', 'Enterprise gifting, measured',
    'How Meridian Hotels turned 300 VIP thank-yous into archived, replenishable gifts.',
    '2026-07-15T08:00:00.000Z', 'scheduled', 'B2B · Wen', '/cases/meridian-hotels-vip', '',
    '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  ),
  (
    'cal_004', 'WeChat', '共创池月报 · 七月',
    '「秩序 03:00」达成 64/100，城市雾气·重庆篇开启投票。',
    '2026-07-05T10:00:00.000Z', 'published', 'Community · Lin', '/co-create', 'ccp_001',
    '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  )
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- orders (4)
-- ----------------------------------------------------------------------------
insert into orders (id, user_id, order_type, title, amount, currency, payment_method, status, reference, created_at, updated_at) values
  ('ord_001', 'usr_member', 'membership',         'Core Sequence Pro · quarterly',                        128,   'CNY', 'wechat_pay',       'test_mode',          'SEQ-PRO-Q3',    '2026-06-18T10:00:00.000Z', '2026-06-18T10:00:00.000Z'),
  ('ord_002', 'usr_lite',   'membership',         'Core Sequence Lite · monthly',                         19,    'CNY', 'alipay',           'test_mode',          'SEQ-LITE-M7',   '2026-06-18T10:00:00.000Z', '2026-06-18T10:00:00.000Z'),
  ('ord_003', 'usr_member', 'physical_casting',   'Midnight Rebuild · casting deposit (12 bottles)',      1440,  'CNY', 'manual_quote',     'awaiting_concierge', 'ZX-Q-2026-121', '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'),
  ('ord_004', 'usr_ent',    'enterprise_project', 'Meridian VIP Appreciation · project deposit',          40500, 'CNY', 'offline_transfer', 'awaiting_concierge', 'ZX-Q-2026-118', '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z')
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- concierge_leads (2)
-- ----------------------------------------------------------------------------
insert into concierge_leads (id, user_id, name, organization, contact, channel, scenario, budget, status, notes, created_at, updated_at) values
  (
    'led_001', 'usr_ent', 'Chen Wei', 'Meridian Hotels Group',
    'gifting@meridian-hotels.example', 'maison',
    'Year-end VIP client appreciation, 300 units, three cities',
    '135,000 RMB', 'quoting',
    'Wants sample path before 09-15; invoice required.',
    '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  ),
  (
    'led_002', null, 'Studio Hyphen', null,
    'hello@studiohyphen.example', 'trade',
    'Brand collaboration — city souvenir box for a design week',
    'To be scoped', 'new', null,
    '2026-06-29T10:00:00.000Z', '2026-06-29T10:00:00.000Z'
  )
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- case_studies (4)
-- ----------------------------------------------------------------------------
insert into case_studies (id, slug, title, title_zh, category, client_type, summary, story, outcome, emotion_tags, featured, created_at) values
  (
    'cas_001', 'meridian-hotels-vip',
    'Meridian Hotels: 300 VIP thank-yous, archived',
    '美芮酒店集团：300 份贵宾致谢，全部入档',
    'Enterprise gifting', 'Hotel group',
    'A year-end appreciation program for top-tier guests: aged huangjiu blend, amber-hinoki room mist, embossed monogram cases — each unit bound to a Reserve certificate.',
    '[
      "Meridian''s gifting team arrived with a scenario, not a spec: thank the 300 guests who define the brand, in three cities, without repeating last year''s hampers.",
      "The AI concierge collected scenario, tiering, and budget, then produced three proposal versions. The human concierge refined version two with the supply chain: an aged huangjiu blend with a citrus-peel finish, paired with an amber-hinoki room mist.",
      "Every unit carries a serial and QR-bound Reserve certificate, so a guest who scans their bottle sees the story written for them — and Meridian can offer replenishment next season."
    ]'::jsonb,
    '300 units delivered across three cities; 41% of recipients scanned their certificate within one month; replenishment window opened for spring.',
    array['gratitude','prestige','continuity'], true, '2026-06-29T10:00:00.000Z'
  ),
  (
    'cas_002', 'order-0300-co-creation',
    'Order 03:00 — a co-created bottle for night rebuilders',
    '秩序 03:00 —— 深夜重建者的共创之瓶',
    'Co-creation', 'Community project',
    'A member''s late-night draft became a 100-bottle co-creation run with 38 supporters, founder engraving, and a label that reads like a manifesto.',
    '[
      "It started as a personal draft: smoked plum, oolong, and one line of label copy written at 3 a.m.",
      "Published to the co-creation pool, the concept crossed the 10-person threshold in two days and unlocked flavor-direction review at 100 bottles.",
      "The founder edition carries an engraved meridian ring; every participant''s bottle binds to a shared batch archive in Reserve."
    ]'::jsonb,
    '64 of 100 bottles reserved in the first month; flavor review passed; production window scheduled.',
    array['night','rebuild','order'], true, '2026-06-29T10:00:00.000Z'
  ),
  (
    'cas_003', 'white-interval-anniversary',
    'White Interval: an anniversary in white florals',
    '白之间隙：白色花香里的纪念日',
    'Personal bespoke', 'Individual member',
    'A restrained anniversary fragrance direction — gardenia and white tea over sandalwood — delivered as a bottle, a card, and a private Reserve record.',
    '[
      "The brief was one sentence: ''S likes white florals and very quiet design.''",
      "The concierge proposed three names and one line of copy — ''Some light does not flare. It stays.'' — and the member kept everything.",
      "The object lives privately in Reserve; the replenishment entry means the eleventh year can be poured again on the twelfth."
    ]'::jsonb,
    'Delivered as a single bespoke unit with engraved base ring; private archive with aftercare active.',
    array['devotion','calm'], false, '2026-06-29T10:00:00.000Z'
  ),
  (
    'cas_004', 'exam-season-survival-kit',
    'Exam Season Survival Kit: zero-proof, full morale',
    '考试季生存补给：零酒精，满士气',
    'Emotional supply', 'Student community',
    'A zero-proof sparkling ration with sticker labels and an emotional card pack, co-created by 41 students for study groups.',
    '[
      "No alcohol, all supply: citrus, sea salt, and a card pack that says the quiet part out loud.",
      "The project cleared minor-safety review as a zero-proof product and reached its 50-unit threshold in nine days.",
      "Every kit ships with a digital mark; the founder''s name sits in the colophon."
    ]'::jsonb,
    '41 supporters at threshold; label co-creation round completed; delivery before finals week.',
    array['persistence','humor'], false, '2026-06-29T10:00:00.000Z'
  )
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- blog_posts (4)
-- ----------------------------------------------------------------------------
insert into blog_posts (id, slug, title, title_zh, excerpt, body, category, author, published_at, featured) values
  (
    'pst_001', 'why-objects-before-carts',
    'Why ZOTAIX asks you to create an object before buying anything',
    '为什么 ZOTAIX 让你先创造对象，而不是先加购物车',
    'The platform''s first principle: understanding precedes commerce. Create, save, and share a personalized object first — decide what it becomes later.',
    '[
      "Most gifting platforms open with a catalog. ZOTAIX opens with a question: what are you feeling, and who is it for?",
      "The difference is structural. A catalog can only match you to inventory. A concierge that collects your emotion, recipient, scenario, and budget can generate something that didn''t exist before — a liquid direction, a fragrance direction, a line of label copy that sounds like you.",
      "That generated object is the unit of value on this platform. It can stay digital forever: saved to your archive, shared as an emotional card, published to the co-creation pool. Or it can become physical: a small-batch casting on a standard base liquid, a premium gift with human concierge confirmation, an enterprise program with a quotation and a sample path.",
      "Commerce arrives at the end of understanding, not the beginning. That is why there is no cart on the homepage."
    ]'::jsonb,
    'Platform', 'ZOTAIX Editorial', '2026-06-10T08:00:00.000Z', true
  ),
  (
    'pst_002', 'small-batch-honesty',
    'Small-batch honesty: what single-bottle customization really means',
    '小批量的诚实：单瓶定制到底意味着什么',
    'Standard base liquid plus personalized expression — and why we will never promise a new formula for every bottle.',
    '[
      "A single customized bottle does not mean a distillery reinvents its process for you. It means a standard, quality-controlled base liquid carries your expression: your label copy, your visual style, your serial number, your story card, your QR-bound archive.",
      "This is the honest boundary of small-order customization, and we print it on the platform: AI-generated results are creative proposals; real products require human confirmation, supply-chain confirmation, age and region compliance checks, and final quotation.",
      "Larger runs unlock deeper customization in stages — 50 bottles open label and gift-box theming, 100 bottles open flavor-direction review, 300 bottles open enterprise gifting review. Depth scales with commitment, and every step is reviewed by humans."
    ]'::jsonb,
    'Compliance', 'ZOTAIX Editorial', '2026-06-17T08:00:00.000Z', true
  ),
  (
    'pst_003', 'reserve-is-the-product',
    'Reserve is the product: on giving objects an identity',
    '档案即产品：为对象赋予身份',
    'QR-bound certificates, aftercare, replenishment — why the archive outlasts the bottle.',
    '[
      "A bottle empties. A fragrance fades. The record of why they existed — who they were for, what they said, which version won — is the part that lasts.",
      "Reserve assigns every object a ZOTAIX ID, a QR/NFC binding, and a certificate page. Public records become shareable pages; private records stay sealed. Aftercare and replenishment attach to the record, not the object, which is how an anniversary bottle becomes an annual ritual.",
      "When you scan a ZOTAIX QR code, you are not reading a product page. You are reading a moment that someone chose to keep."
    ]'::jsonb,
    'Product', 'ZOTAIX Editorial', '2026-06-24T08:00:00.000Z', false
  ),
  (
    'pst_004', 'mbti-is-a-brush-not-a-box',
    'MBTI is a brush, not a box',
    'MBTI 是画笔，不是盒子',
    'How ZOTAIX uses self-expression tags — and how it refuses to use them.',
    '[
      "MBTI, zodiac signs, and blood types appear in your ZOTAIX profile for one reason: they are vocabularies people already use to describe themselves.",
      "The platform treats them as style signals — a preference for how you like to be spoken to — never as diagnosis, prediction, or gatekeeping. They influence tone and imagery, not access or price.",
      "Every tag is optional, editable, deletable, and excluded from public display by default. These details do not define who you are; they help ZOTAIX understand your preferred way of expression."
    ]'::jsonb,
    'Design', 'ZOTAIX Editorial', '2026-06-28T08:00:00.000Z', false
  )
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- cms_blocks (4)
-- ----------------------------------------------------------------------------
insert into cms_blocks (id, key, title, content, page, enabled, updated_at) values
  ('cms_001', 'home.hero.badge',        'Homepage hero badge',            'AI Concierge Customization Platform', '/', true, '2026-06-29T10:00:00.000Z'),
  ('cms_002', 'home.announcement',      'Homepage announcement bar',      'Co-creation ''Order 03:00'' is 64% reserved — founder engraving closes at 100 bottles.', '/', true, '2026-06-29T10:00:00.000Z'),
  ('cms_003', 'maison.concierge.hours', 'Concierge working hours',        'Human concierge replies within one business day, 10:00–19:00 CST.', '/maison', true, '2026-06-29T10:00:00.000Z'),
  ('cms_004', 'download.beta.note',     'Download page distribution note','Store listings are configured by the operations team; until a store link is set, the buttons below explain how to install the web app.', '/download', true, '2026-06-29T10:00:00.000Z')
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- legal_docs (12)
-- ----------------------------------------------------------------------------
insert into legal_docs (id, slug, title, version, effective_date, updated_at) values
  ('lgl_terms',      'terms',      'User Terms',                   '2.1', '2026-05-01', '2026-06-29T10:00:00.000Z'),
  ('lgl_privacy',    'privacy',    'Privacy Policy',               '2.3', '2026-05-01', '2026-06-29T10:00:00.000Z'),
  ('lgl_cookies',    'cookies',    'Cookie Policy',                '1.2', '2026-05-01', '2026-06-29T10:00:00.000Z'),
  ('lgl_ai',         'ai',         'AI Generated Content Notice',  '1.4', '2026-05-01', '2026-06-29T10:00:00.000Z'),
  ('lgl_alcohol',    'alcohol',    'Alcohol Compliance Notice',    '1.5', '2026-05-01', '2026-06-29T10:00:00.000Z'),
  ('lgl_minors',     'minors',     'Minor Protection Notice',      '1.1', '2026-05-01', '2026-06-29T10:00:00.000Z'),
  ('lgl_membership', 'membership', 'Membership Service Agreement', '1.3', '2026-05-01', '2026-06-29T10:00:00.000Z'),
  ('lgl_cocreate',   'co-create',  'Co-Creation Pool Rules',       '1.2', '2026-05-01', '2026-06-29T10:00:00.000Z'),
  ('lgl_trade',      'trade',      'Trade Creative Market Rules',  '1.2', '2026-05-01', '2026-06-29T10:00:00.000Z'),
  ('lgl_reserve',    'reserve',    'Reserve Archive Rules',        '1.1', '2026-05-01', '2026-06-29T10:00:00.000Z'),
  ('lgl_app',        'app',        'App Privacy Notice',           '1.0', '2026-05-01', '2026-06-29T10:00:00.000Z'),
  ('lgl_contact',    'contact',    'Contact Us',                   '1.0', '2026-05-01', '2026-06-29T10:00:00.000Z')
on conflict (id) do nothing;

-- ----------------------------------------------------------------------------
-- platform_settings (1)
-- ----------------------------------------------------------------------------
insert into platform_settings (
  id, site_name, brand_line_en, brand_line_zh, guest_daily_chat,
  free_daily_chat, lite_daily_chat, pro_daily_chat, lite_monthly_proposals,
  pro_monthly_proposals, lite_price_month, lite_price_quarter, pro_price_month,
  pro_price_quarter, co_create_public_threshold, co_create_review_threshold,
  co_create_label_threshold, co_create_flavor_threshold,
  co_create_enterprise_threshold, co_create_supply_threshold,
  co_create_partner_threshold, age_gate_enabled, updated_at
) values (
  'set_001', 'ZOTAIX',
  'ZOTAIX is an AI concierge platform that turns emotions, relationships, scenarios, and budgets into bespoke spirits, fragrance directions, bottle design, gifting systems, and digital identity records.',
  'ZOTAIX 卓序是一个 AI 礼宾式定制平台，用 AI 将人的情绪、关系、场景和预算，转化为可确认、可报价、可交付的酒饮、香氛、瓶身、包装与礼赠方案。',
  3, 3, 15, 50, 20, 80, 19, 49, 49, 128,
  10, 30, 50, 100, 300, 500, 1000,
  true, '2026-06-29T10:00:00.000Z'
)
on conflict (id) do nothing;
