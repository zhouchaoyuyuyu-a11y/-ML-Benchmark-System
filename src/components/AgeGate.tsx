"use client";

import { useEffect, useState } from "react";

/** Age confirmation modal for alcohol-related pages. Remembered per browser. */
export default function AgeGate({ zh = false }: { zh?: boolean }) {
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined" && !localStorage.getItem("zx_age_confirmed")) {
      setShow(true);
    }
  }, []);

  if (!show) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/85 p-4 backdrop-blur-sm">
      <div className="zx-card w-full max-w-md p-6 text-center">
        <p className="font-display text-xl tracking-[0.18em] text-porcelain">ZOTAIX</p>
        <div className="zx-meridian my-4" />
        <p className="font-display text-lg text-porcelain">
          {zh ? "本页面包含酒类相关内容" : "This page contains alcohol-related content"}
        </p>
        <p className="mt-3 text-sm leading-relaxed text-mist">
          {zh
            ? "请确认你已达到所在地区的法定饮酒年龄。ZOTAIX 不向未成年人推广或销售酒类产品。"
            : "Please confirm you are of legal drinking age in your region. ZOTAIX does not promote or sell alcohol to minors."}
        </p>
        <div className="mt-6 flex flex-col gap-2 sm:flex-row sm:justify-center">
          <button
            onClick={() => {
              localStorage.setItem("zx_age_confirmed", "1");
              setShow(false);
            }}
            className="rounded-md bg-gold px-6 py-2.5 text-sm font-medium text-ink transition-colors hover:bg-gold-deep hover:text-porcelain"
          >
            {zh ? "我已成年，进入" : "I am of legal age — enter"}
          </button>
          <a
            href="/supply"
            className="rounded-md border border-hairline px-6 py-2.5 text-sm text-mist transition-colors hover:text-porcelain"
          >
            {zh ? "查看零酒精补给" : "See zero-proof supplies"}
          </a>
        </div>
        <p className="mt-4 text-xs text-mist">
          {zh ? "理性饮酒 · 实体交付需年龄与地区合规审核" : "Drink responsibly · Physical delivery requires age & region compliance checks"}
        </p>
      </div>
    </div>
  );
}
