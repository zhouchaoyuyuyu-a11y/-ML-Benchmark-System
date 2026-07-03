import { ImageResponse } from "next/og";

export const runtime = "edge";

/** Shareable emotional card image (WeChat / social friendly 3:4). */
export async function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const copy = (searchParams.get("copy") ?? "Begin anywhere. The order will follow.").slice(0, 120);
  const mark = (searchParams.get("mark") ?? "Prelude Mark").slice(0, 60);
  const keywords = (searchParams.get("keywords") ?? "open signal").slice(0, 80);

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: 64,
          background: "linear-gradient(160deg, #0a0c11 0%, #12151d 55%, #26224a 130%)",
          color: "#ece9e2",
          fontFamily: "Georgia, serif",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ fontSize: 26, letterSpacing: 6 }}>ZOTAIX</div>
          <div style={{ fontSize: 18, color: "#8b93ff", fontFamily: "sans-serif" }}>◈ Emotional Card</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
          <div style={{ fontSize: 20, color: "#c8a962", letterSpacing: 3, fontFamily: "sans-serif" }}>{keywords}</div>
          <div style={{ fontSize: 46, lineHeight: 1.3, fontStyle: "italic", maxWidth: 760 }}>“{copy}”</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ height: 2, background: "linear-gradient(90deg, transparent, #c8a962, transparent)" }} />
          <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "sans-serif" }}>
            <div style={{ fontSize: 18, color: "#9aa0ad" }}>{mark}</div>
            <div style={{ fontSize: 18, color: "#9aa0ad" }}>zotaix · 卓序</div>
          </div>
        </div>
      </div>
    ),
    { width: 900, height: 1200 }
  );
}
