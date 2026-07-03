// Runtime data layer. When DATABASE_URL is configured, deploy db/schema.sql +
// db/seed.sql to Postgres/Supabase and swap the adapter; without it, this
// seeded in-memory store keeps every feature functional (state is per server
// instance and resets on redeploy — see README "Data layer").

import type {
  AiUsageLog,
  AppConfig,
  BlogPost,
  CaseStudy,
  CmsBlock,
  ConciergeLead,
  ContentCalendarItem,
  Conversation,
  CoCreationMember,
  CoCreationProject,
  DesignVersion,
  LegalDoc,
  Membership,
  Message,
  ModerationLog,
  ObjectDraft,
  Order,
  PlatformSettings,
  RelationshipProfile,
  ReserveRecord,
  SocialAccount,
  TradeRequest,
  User,
  UserProfile,
  WechatConfig,
} from "./types";
import {
  seedAppConfig,
  seedCalendar,
  seedCases,
  seedCms,
  seedConversations,
  seedDrafts,
  seedLeads,
  seedLegal,
  seedMemberships,
  seedMessages,
  seedModeration,
  seedOrders,
  seedPosts,
  seedProfiles,
  seedProjectMembers,
  seedProjects,
  seedRelationships,
  seedReserve,
  seedSettings,
  seedSocial,
  seedTrade,
  seedUsage,
  seedUsers,
  seedVersions,
  seedWechat,
} from "./seed";

export interface Database {
  users: User[];
  user_profiles: UserProfile[];
  relationship_profiles: RelationshipProfile[];
  conversations: Conversation[];
  messages: Message[];
  ai_usage_logs: AiUsageLog[];
  object_drafts: ObjectDraft[];
  design_versions: DesignVersion[];
  memberships: Membership[];
  co_creation_projects: CoCreationProject[];
  co_creation_members: CoCreationMember[];
  reserve_records: ReserveRecord[];
  trade_requests: TradeRequest[];
  moderation_logs: ModerationLog[];
  social_accounts: SocialAccount[];
  wechat_config: WechatConfig;
  app_config: AppConfig;
  content_calendar: ContentCalendarItem[];
  orders: Order[];
  concierge_leads: ConciergeLead[];
  case_studies: CaseStudy[];
  blog_posts: BlogPost[];
  cms_blocks: CmsBlock[];
  legal_docs: LegalDoc[];
  settings: PlatformSettings;
}

function clone<T>(v: T): T {
  return JSON.parse(JSON.stringify(v));
}

function freshDatabase(): Database {
  return clone({
    users: seedUsers,
    user_profiles: seedProfiles,
    relationship_profiles: seedRelationships,
    conversations: seedConversations,
    messages: seedMessages,
    ai_usage_logs: seedUsage,
    object_drafts: seedDrafts,
    design_versions: seedVersions,
    memberships: seedMemberships,
    co_creation_projects: seedProjects,
    co_creation_members: seedProjectMembers,
    reserve_records: seedReserve,
    trade_requests: seedTrade,
    moderation_logs: seedModeration,
    social_accounts: seedSocial,
    wechat_config: seedWechat,
    app_config: seedAppConfig,
    content_calendar: seedCalendar,
    orders: seedOrders,
    concierge_leads: seedLeads,
    case_studies: seedCases,
    blog_posts: seedPosts,
    cms_blocks: seedCms,
    legal_docs: seedLegal,
    settings: seedSettings,
  });
}

const globalRef = globalThis as unknown as { __zotaixDb?: Database };

export function db(): Database {
  if (!globalRef.__zotaixDb) globalRef.__zotaixDb = freshDatabase();
  return globalRef.__zotaixDb;
}

export function dataMode(): "memory" | "database" {
  return process.env.DATABASE_URL ? "database" : "memory";
}

let counter = 1000;
export function newId(prefix: string): string {
  counter += 1;
  const rand = Math.random().toString(36).slice(2, 8);
  return `${prefix}_${counter.toString(36)}${rand}`;
}

export function now(): string {
  return new Date().toISOString();
}

export function newZotaixId(): string {
  const d = new Date();
  const ymd = `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, "0")}${String(d.getUTCDate()).padStart(2, "0")}`;
  const serial = String(db().reserve_records.length + 1).padStart(4, "0");
  return `ZX-${ymd.slice(0, 4)}-${ymd.slice(4)}-${serial}`;
}

export function versionHash(payload: unknown): string {
  const s = JSON.stringify(payload);
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return `zx-${(h >>> 0).toString(16).padStart(8, "0")}`;
}
