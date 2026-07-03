import { NextResponse } from "next/server";
import { aiProvider } from "@/lib/config";
import { dataMode } from "@/lib/store";

export async function GET() {
  return NextResponse.json({
    ok: true,
    service: "zotaix-web",
    dataMode: dataMode(),
    aiProvider: aiProvider(),
    time: new Date().toISOString(),
  });
}
