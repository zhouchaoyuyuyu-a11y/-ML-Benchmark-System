import type { Metadata } from "next";
import QRCodeBox from "@/components/QRCodeBox";
import { ButtonLink, Card, Notice, PageHero, Section, SectionHeader, Tag } from "@/components/ui";
import { complianceNotice } from "@/lib/copy";
import { getLocale } from "@/lib/locale";
import { pageMetadata } from "@/lib/seo";
import { db } from "@/lib/store";
import StudioClient from "./StudioClient";

export const metadata: Metadata = pageMetadata({
  title: "Studio — visual preview for bottles, labels, packaging, and moods",
  description:
    "The ZOTAIX Studio is the visual layer of the customization chain: a rotating 3D bottle configurator, live label copy, packaging previews, fragrance mood boards, emotional cards, and digital badge medallions.",
  path: "/studio",
});

export default async function StudioPage() {
  const locale = await getLocale();
  const zh = locale === "zh";
  const data = db();

  const castings = data.design_versions.slice(0, 3);
  const draftTitle = (id: string) =>
    data.object_drafts.find((d) => d.id === id)?.title ?? (zh ? "已归档草案" : "Archived draft");

  const layers = [
    {
      icon: "◇",
      en: "Bottle form",
      zh: "瓶身形态",
      dEn: "Three silhouettes — tall cylinder, wide decanter, slender flask — rendered as a slowly rotating casting preview.",
      dZh: "三种轮廓——高柱瓶、宽肩醒酒瓶、细长瓶——以缓慢旋转的铸造预览呈现。",
    },
    {
      icon: "▤",
      en: "Glass & accent",
      zh: "玻璃与点缀",
      dEn: "Ink black, porcelain white, ink blue, or frosted glass, finished with gold, silver, or supply-violet accents.",
      dZh: "墨黑、瓷白、墨蓝或磨砂玻璃，配鎏金、银或补给紫点缀。",
    },
    {
      icon: "✎",
      en: "Label & copy",
      zh: "标签与文案",
      dEn: "Vertical serif, horizontal band, or sticker labels — your line of copy renders on the bottle as you type.",
      dZh: "竖排衬线、横向腰封或贴纸标签——你的文案边打边上瓶。",
    },
    {
      icon: "▣",
      en: "Packaging",
      zh: "包装",
      dEn: "Slipcase, magnetic box, or kraft sleeve, each previewed with materials and closure details.",
      dZh: "抽拉函套、磁吸礼盒或牛皮纸套，均附材质与开合细节预览。",
    },
    {
      icon: "❖",
      en: "Fragrance mood",
      zh: "香氛情绪",
      dEn: "Six mood boards, each a gradient swatch with a top / heart / base note pyramid.",
      dZh: "六块情绪板，每块都是一组渐变色样与前 / 中 / 尾调金字塔。",
    },
    {
      icon: "◈",
      en: "Cards & badges",
      zh: "卡片与徽章",
      dEn: "Emotional cards rendered as shareable 3:4 images, and digital marks previewed as bordered medallions.",
      dZh: "情绪卡片生成可分享的 3:4 图像，数字印记以描边徽章形式预览。",
    },
  ];

  const badges = [
    {
      glyph: "◈",
      name: zh ? "纪念封印 · No. 011" : "Anniversary Seal · No. 011",
      serial: "ZX-MARK-0011",
      dEn: "A private mark bound to the White Interval fragrance direction.",
      dZh: "绑定「白之间隙」香氛方向的私人印记。",
    },
    {
      glyph: "⬡",
      name: zh ? "创始者印记 · 秩序 03:00" : "Founder Mark · Order 03:00",
      serial: "ZX-FND-0064",
      dEn: "Issued to founders of the 100-bottle co-creation run.",
      dZh: "授予百瓶共创项目创始成员的印记。",
    },
    {
      glyph: "✶",
      name: zh ? "夜读补给 · 数字印记" : "Night Study Ration · Digital Mark",
      serial: "QR-ZX-C2D40F11",
      dEn: "A public supply-line badge from an exam-season label.",
      dZh: "来自考试季标签的公开补给线徽章。",
    },
  ];

  return (
    <>
      <PageHero
        eyebrow={zh ? "Studio · 视觉预览层" : "Studio · Visual Preview Layer"}
        title={zh ? "在成为实物之前，先看见它" : "See the object before it exists"}
        description={
          zh
            ? "Studio 是 ZOTAIX 定制链的视觉层：瓶身、标签、包装、香氛情绪、情绪卡片与数字徽章，全部在浏览器里实时预览。你在这里做的每一个选择都可以存为草案——是否成为实体，之后再决定。"
            : "The Studio is the visual layer of the ZOTAIX chain: bottle, label, packaging, fragrance mood, emotional cards, and digital badges, all previewed live in the browser. Every choice you make here can be saved as a draft — whether it becomes physical is a separate decision, made on your terms."
        }
      >
        <ButtonLink href="#configurator" variant="gold">
          {zh ? "打开瓶身配置器" : "Open the bottle configurator"}
        </ButtonLink>
        <ButtonLink href="/forge" variant="outline">
          {zh ? "先在 Forge 生成方向" : "Generate a direction in Forge first"}
        </ButtonLink>
        <ButtonLink href="/design" variant="outline">
          {zh ? "查看我的草案与版本" : "My drafts & versions in Design"}
        </ButtonLink>
      </PageHero>

      {/* Layer overview */}
      <Section className="py-12 sm:py-16">
        <SectionHeader
          eyebrow={zh ? "六个预览层" : "Six preview layers"}
          title={zh ? "一个对象的全部可见面" : "Every visible face of one object"}
          description={
            zh
              ? "从 Forge 带来一份提案，或者直接从零开始搭——Studio 把方向变成可以被看见、被分享、被存档的形态。"
              : "Bring a proposal from the Forge or build from zero — the Studio turns a direction into something that can be seen, shared, and archived."
          }
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {layers.map((l) => (
            <Card key={l.en} className="h-full">
              <span className="text-lg text-gold">{l.icon}</span>
              <p className="font-display mt-2 text-base text-porcelain">{zh ? l.zh : l.en}</p>
              <p className="mt-1.5 text-sm leading-relaxed text-mist">{zh ? l.dZh : l.dEn}</p>
            </Card>
          ))}
        </div>
      </Section>

      {/* Configurator */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section id="configurator" className="py-12 sm:py-16">
          <SectionHeader
            eyebrow={zh ? "瓶身配置器" : "The bottle configurator"}
            title={zh ? "旋转的铸造预览，实时响应每个选择" : "A rotating casting preview that answers every choice"}
            description={
              zh
                ? "两块交叉的全息剖面构成瓶身，随时间缓慢旋转。改变瓶型、玻璃、点缀、标签或文案，预览立即更新。"
                : "Two crossed holographic planes form the bottle, rotating slowly on its axis. Change the shape, glass, accent, label, or copy, and the preview updates instantly."
            }
          />
          <div className="mt-8">
            <StudioClient zh={zh} />
          </div>
        </Section>
      </div>

      {/* Recent castings */}
      <Section className="py-12 sm:py-16">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <SectionHeader
            eyebrow={zh ? "最近的铸造" : "Recent castings"}
            title={zh ? "从 Studio 走进档案的版本" : "Versions that left the Studio for the archive"}
            description={
              zh
                ? "每个保存的方向都会获得一个版本哈希与身份码。扫码即达版本页——这是 Reserve 与 Trade 引用设计的方式。"
                : "Every saved direction receives a version hash and an identity code. The code resolves to the version page — this is how Reserve and Trade cite a design."
            }
          />
          <ButtonLink href="/design" variant="outline">
            {zh ? "全部版本在 Design" : "All versions in Design"}
          </ButtonLink>
        </div>
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {castings.map((v) => (
            <Card key={v.id} className="flex h-full flex-col gap-4">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate text-xs uppercase tracking-wider text-gold">{draftTitle(v.object_draft_id)}</p>
                  <p className="font-display mt-1 text-base text-porcelain">{v.version_name}</p>
                  <p className="mt-1 text-xs text-mist">{v.created_at.slice(0, 10)}</p>
                  <code className="mt-2 inline-block rounded bg-veil px-2 py-0.5 font-mono text-xs text-gold">
                    {v.version_hash}
                  </code>
                </div>
                <QRCodeBox seed={v.version_hash} size={96} />
              </div>
              <div className="space-y-1.5 border-t border-hairline pt-3">
                {Object.entries(v.design_payload).map(([key, value]) => (
                  <p key={key} className="text-xs leading-relaxed text-mist">
                    <span className="uppercase tracking-wider text-porcelain/70">{key}</span> · {value}
                  </p>
                ))}
              </div>
            </Card>
          ))}
        </div>
      </Section>

      {/* Digital badges */}
      <div className="border-y border-hairline bg-obsidian/40">
        <Section className="py-12 sm:py-16">
          <SectionHeader
            eyebrow={zh ? "数字徽章预览" : "Digital badge preview"}
            title={zh ? "有些对象永远是数字的——这也很好" : "Some objects stay digital forever — that is a fine ending"}
            description={
              zh
                ? "数字印记是 ZOTAIX 最轻的对象形态：一枚描边徽章、一个序列号、一条档案记录。它们可以被赠送、被展示、被扫码。"
                : "Digital marks are the lightest object on ZOTAIX: a bordered medallion, a serial, an archive record. They can be gifted, displayed, and scanned."
            }
          />
          <div className="mt-10 grid gap-8 sm:grid-cols-3">
            {badges.map((b) => (
              <div key={b.serial} className="flex flex-col items-center gap-4 text-center">
                <div className="flex h-32 w-32 items-center justify-center rounded-full border-2 border-gold/40 bg-obsidian shadow-[inset_0_0_28px_rgba(200,169,98,0.16)]">
                  <div className="flex h-24 w-24 flex-col items-center justify-center gap-1 rounded-full border border-gold/30">
                    <span className="text-2xl text-gold">{b.glyph}</span>
                    <span className="text-[10px] uppercase tracking-[0.2em] text-mist">ZOTAIX</span>
                  </div>
                </div>
                <div>
                  <p className="font-display text-sm text-porcelain">{b.name}</p>
                  <p className="mt-0.5 font-mono text-xs text-gold/80">{b.serial}</p>
                  <p className="mt-1.5 text-xs leading-relaxed text-mist">{zh ? b.dZh : b.dEn}</p>
                </div>
              </div>
            ))}
          </div>
        </Section>
      </div>

      {/* Compliance + next steps */}
      <Section className="py-12 sm:py-16">
        <Notice tone="gold">{zh ? complianceNotice.zh : complianceNotice.en}</Notice>
        <div className="mt-8 flex flex-wrap items-center gap-3">
          <Tag tone="gold">{zh ? "下一步" : "Next"}</Tag>
          <ButtonLink href="/design" variant="gold">
            {zh ? "整理版本 · Design" : "Refine versions in Design"}
          </ButtonLink>
          <ButtonLink href="/trade" variant="outline">
            {zh ? "申请铸造报价 · Trade" : "Request a casting quote in Trade"}
          </ButtonLink>
          <ButtonLink href="/reserve" variant="outline">
            {zh ? "入档 · Reserve" : "Archive it in Reserve"}
          </ButtonLink>
        </div>
      </Section>
    </>
  );
}
