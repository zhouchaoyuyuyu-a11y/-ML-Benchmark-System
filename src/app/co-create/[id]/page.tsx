import Link from "next/link";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { Card, Notice, Section, StatusPill, Tag } from "@/components/ui";
import JoinClient from "./JoinClient";
import { getSessionUser } from "@/lib/auth";
import { complianceNotice, pick } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";

export async function generateMetadata({ params }: { params: Promise<{ id: string }> }): Promise<Metadata> {
  const { id } = await params;
  const project = db().co_creation_projects.find(
    (p) => p.id === id && p.public_visible && p.review_status === "approved"
  );
  if (!project) {
    return pageMetadata({
      title: "Co-creation project",
      description: "A collective casting project in the ZOTAIX Co-Creation Pool.",
      path: `/co-create/${id}`,
    });
  }
  return pageMetadata({
    title: `${project.title} — co-creation project`,
    description: project.concept.slice(0, 160),
    path: `/co-create/${id}`,
  });
}

export default async function CoCreateDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();
  const user = await getSessionUser();

  const project = data.co_creation_projects.find((p) => p.id === id);
  const isCreator = !!project && !!user && user.id === project.creator_user_id;
  const publiclyVisible = !!project && project.public_visible && project.review_status === "approved";
  if (!project || (!publiclyVisible && !isCreator)) notFound();

  const s = data.settings;
  const membersCount = data.co_creation_members.filter((m) => m.project_id === project.id).length;
  const founder = data.users.find((u) => u.id === project.creator_user_id);

  const thresholds = [
    {
      value: s.co_create_public_threshold,
      crossed: project.supporters >= s.co_create_public_threshold,
      en: `${s.co_create_public_threshold} supporters — public project page`,
      zh: `${s.co_create_public_threshold} 位支持者 —— 公开项目页`,
    },
    {
      value: s.co_create_review_threshold,
      crossed: project.current_quantity >= s.co_create_review_threshold,
      en: `${s.co_create_review_threshold} units — deepened platform review`,
      zh: `${s.co_create_review_threshold} 份 —— 平台深度评审`,
    },
    {
      value: s.co_create_label_threshold,
      crossed: project.current_quantity >= s.co_create_label_threshold,
      en: `${s.co_create_label_threshold} units — label & gift-box co-creation`,
      zh: `${s.co_create_label_threshold} 份 —— 标签与礼盒共创`,
    },
    {
      value: s.co_create_flavor_threshold,
      crossed: project.current_quantity >= s.co_create_flavor_threshold,
      en: `${s.co_create_flavor_threshold} units — flavor-direction review`,
      zh: `${s.co_create_flavor_threshold} 份 —— 风味方向评审`,
    },
    {
      value: s.co_create_enterprise_threshold,
      crossed: project.current_quantity >= s.co_create_enterprise_threshold,
      en: `${s.co_create_enterprise_threshold} units — enterprise gifting review`,
      zh: `${s.co_create_enterprise_threshold} 份 —— 企业礼赠评审`,
    },
    {
      value: s.co_create_supply_threshold,
      crossed: project.current_quantity >= s.co_create_supply_threshold,
      en: `${s.co_create_supply_threshold}+ units — supply-chain scheduling`,
      zh: `${s.co_create_supply_threshold}+ 份 —— 供应链排产`,
    },
    {
      value: s.co_create_partner_threshold,
      crossed: project.current_quantity >= s.co_create_partner_threshold,
      en: `${s.co_create_partner_threshold}+ units — brand partnership track`,
      zh: `${s.co_create_partner_threshold}+ 份 —— 品牌合作通道`,
    },
  ];
  const crossedCount = thresholds.filter((t) => t.crossed).length;

  return (
    <>
      <div className="zx-grid-bg border-b border-hairline">
        <Section className="py-12 sm:py-16">
          <Link href="/co-create" className="text-xs text-mist transition-colors hover:text-gold">
            ← {zh ? "返回共创池" : "Back to the Co-Creation Pool"}
          </Link>
          <div className="mt-4 flex flex-wrap items-center gap-2">
            <Tag tone="gold">{project.product_type}</Tag>
            <StatusPill status={project.status} />
            {!publiclyVisible && <StatusPill status={project.review_status} />}
            {project.emotion_tags.map((t) => (
              <Tag key={t}>{t}</Tag>
            ))}
          </div>
          <h1 className="font-display mt-4 max-w-4xl text-3xl leading-tight text-porcelain sm:text-4xl">
            {project.title}
          </h1>
          <p className="mt-3 text-sm text-mist">
            {zh ? "发起人：" : "Founded by "}
            <span className="text-porcelain">{founder?.nickname ?? "ZOTAIX member"}</span>
            {" · "}
            {zh ? "创建于 " : "created "}
            {project.created_at.slice(0, 10)}
          </p>
        </Section>
      </div>

      <Section className="py-10 sm:py-14">
        <div className="grid gap-8 lg:grid-cols-[1fr_360px]">
          {/* Main column */}
          <div className="space-y-6">
            {!publiclyVisible && (
              <Notice tone="gold" title={zh ? "仅发起人可见" : "Visible to you as the founder"}>
                {zh
                  ? "这个项目仍在平台评审中，通过公开展示审核后才会开放投票与加入。你可以在个人中心跟踪评审状态。"
                  : "This project is still in platform review; voting and joining open once public-display review passes. You can track review status in your profile."}
              </Notice>
            )}

            <Card>
              <p className="font-display text-lg text-porcelain">{zh ? "项目概念" : "The concept"}</p>
              <p className="mt-3 text-sm leading-relaxed text-mist sm:text-base">{project.concept}</p>
            </Card>

            <Card className="border-gold/25">
              <p className="text-xs font-semibold uppercase tracking-[0.25em] text-gold">
                {zh ? "发起人权益" : "Founder benefit"}
              </p>
              <p className="font-display mt-2 text-base text-porcelain">{project.founder_benefit}</p>
              <ul className="mt-4 space-y-2 text-sm leading-relaxed text-mist">
                <li className="flex items-start gap-2.5">
                  <span className="mt-0.5 text-gold">◆</span>
                  <span>
                    {zh
                      ? "发起人权益随项目达成一并交付，写入创始版与专属档案页。"
                      : "Founder rights are delivered with the completed run — written into the founder edition and its exclusive archive page."}
                  </span>
                </li>
                <li className="flex items-start gap-2.5">
                  <span className="mt-0.5 text-gold">▣</span>
                  <span>
                    {zh
                      ? "参与者获得共享批次档案绑定与个人数字印记。"
                      : "Participants receive the shared batch archive binding and a personal digital mark."}
                  </span>
                </li>
              </ul>
            </Card>

            <Card>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="font-display text-lg text-porcelain">{zh ? "门槛清单" : "Threshold checklist"}</p>
                <Tag tone={crossedCount > 0 ? "jade" : "default"}>
                  {crossedCount}/{thresholds.length} {zh ? "已跨越" : "crossed"}
                </Tag>
              </div>
              <ul className="mt-4 space-y-2.5">
                {thresholds.map((t) => (
                  <li key={t.en} className="flex items-start gap-3 text-sm leading-relaxed">
                    <span
                      className={
                        t.crossed
                          ? "mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-jade/50 bg-jade/10 text-xs text-jade"
                          : "mt-0.5 inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-hairline text-xs text-mist"
                      }
                    >
                      {t.crossed ? "✓" : "○"}
                    </span>
                    <span className={t.crossed ? "text-porcelain" : "text-mist"}>{zh ? t.zh : t.en}</span>
                  </li>
                ))}
              </ul>
              <p className="mt-4 text-xs leading-relaxed text-mist">
                {zh
                  ? "第一级门槛按支持者人数计，其余按预订份数计。每一级解锁都伴随一轮人工评审。"
                  : "The first rung counts supporters; the rest count reserved units. Every unlocked stage triggers a round of human review."}
              </p>
            </Card>

            <Notice tone="ember" title={zh ? "审核与合规" : "Review & compliance"}>
              {zh
                ? "本项目接受敏感内容、酒类合规、未成年人保护、版权、可行性、公开展示与交易资格审查。涉及酒精的交付执行年龄与地区审核；预订阶段不收取任何款项。"
                : "This project is reviewed for sensitive content, alcohol compliance, minor safety, copyright, feasibility, public display, and trade eligibility. Alcohol-related delivery runs age and region checks; no payment is taken at the reservation stage."}
            </Notice>
            <Notice tone="gold" title={zh ? "合规声明" : "Compliance"}>
              {pick(locale, complianceNotice)}
            </Notice>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            <JoinClient
              zh={zh}
              projectId={project.id}
              gathering={project.status === "gathering" && publiclyVisible}
              targetQuantity={project.target_quantity}
              initialQuantity={project.current_quantity}
              initialSupporters={project.supporters}
              initialVotes={project.votes}
              membersCount={membersCount}
            />

            <Card>
              <p className="font-display text-base text-porcelain">{zh ? "分享这个项目" : "Share this project"}</p>
              <p className="mt-2 text-xs leading-relaxed text-mist">
                {zh
                  ? "多一位支持者，就离下一级门槛更近一步。"
                  : "One more supporter is one step closer to the next threshold."}
              </p>
              <div className="mt-4 flex flex-col gap-2 text-sm">
                <Link href="/wechat" className="text-gold hover:underline">
                  {zh ? "微信公众号内分享 →" : "Share inside WeChat →"}
                </Link>
                <Link href="/social" className="text-gold hover:underline">
                  {zh ? "全球社媒矩阵 →" : "Global social channels →"}
                </Link>
              </div>
            </Card>
          </div>
        </div>
      </Section>
    </>
  );
}
