import { ImageResponse } from "next/og";

export const runtime = "edge";

/** Dynamic Open Graph image for public pages. */
export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const title = (searchParams.get("title") ?? "ZOTAIX").slice(0, 90);
  const subtitle = (
    searchParams.get("subtitle") ??
    "AI concierge platform · emotions, relationships, scenarios and budgets → bespoke spirits, fragrances, bottles, gifts"
  ).slice(0, 140);

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: 72,
          background: "linear-gradient(135deg, #0a0c11 0%, #12151d 60%, #1c202b 100%)",
          color: "#ece9e2",
          fontFamily: "Georgia, serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div
            style={{
              width: 56,
              height: 56,
              border: "2px solid #c8a962",
              borderRadius: 12,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#c8a962",
              fontSize: 34,
            }}
          >
            Z
          </div>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <div style={{ fontSize: 28, letterSpacing: 6 }}>ZOTAIX</div>
            <div style={{ fontSize: 16, color: "#c8a962", letterSpacing: 8 }}>卓序</div>
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div style={{ fontSize: 58, lineHeight: 1.15, maxWidth: 980 }}>{title}</div>
          <div style={{ fontSize: 24, color: "#9aa0ad", maxWidth: 940, fontFamily: "sans-serif" }}>{subtitle}</div>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ height: 2, flex: 1, background: "linear-gradient(90deg, transparent, #c8a962, transparent)" }} />
        </div>
      </div>
    ),
    { width: 1200, height: 630 }
  );
}
