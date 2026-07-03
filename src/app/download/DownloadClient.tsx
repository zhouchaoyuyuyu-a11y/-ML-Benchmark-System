"use client";

import { useEffect, useState } from "react";
import { Button, Notice } from "@/components/ui";

/* PWA install helper: listens for beforeinstallprompt where the browser
   exposes it, detects the WeChat in-app browser, and walks through
   per-browser install paths. */

interface BeforeInstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed"; platform: string }>;
}

type InstallState = "idle" | "prompting" | "accepted" | "dismissed" | "installed";

export default function DownloadClient({ zh = false }: { zh?: boolean }) {
  const [installEvent, setInstallEvent] = useState<BeforeInstallPromptEvent | null>(null);
  const [installState, setInstallState] = useState<InstallState>("idle");
  const [isWeChat, setIsWeChat] = useState(false);
  const [isStandalone, setIsStandalone] = useState(false);
  const [openGuide, setOpenGuide] = useState<number | null>(0);

  useEffect(() => {
    if (/MicroMessenger/i.test(navigator.userAgent)) setIsWeChat(true);
    if (window.matchMedia && window.matchMedia("(display-mode: standalone)").matches) {
      setIsStandalone(true);
    }
    const onPrompt = (e: Event) => {
      e.preventDefault();
      setInstallEvent(e as BeforeInstallPromptEvent);
    };
    const onInstalled = () => setInstallState("installed");
    window.addEventListener("beforeinstallprompt", onPrompt);
    window.addEventListener("appinstalled", onInstalled);
    return () => {
      window.removeEventListener("beforeinstallprompt", onPrompt);
      window.removeEventListener("appinstalled", onInstalled);
    };
  }, []);

  async function install() {
    if (!installEvent) return;
    setInstallState("prompting");
    await installEvent.prompt();
    const choice = await installEvent.userChoice;
    setInstallState(choice.outcome === "accepted" ? "accepted" : "dismissed");
    if (choice.outcome === "accepted") setInstallEvent(null);
  }

  const guides: { id: string; title: string; badge: string; steps: string[] }[] = [
    {
      id: "ios-safari",
      title: zh ? "iOS Safari · 添加到主屏幕" : "iOS Safari · Add to Home Screen",
      badge: "iPhone / iPad",
      steps: zh
        ? [
            "在 Safari 中打开本页（微信内请先点右上角 ··· 选择「在浏览器中打开」）。",
            "点击底部工具栏的分享按钮（方框加向上箭头）。",
            "在菜单中向下滑动，选择「添加到主屏幕」。",
            "确认名称为 ZOTAIX，点击「添加」。",
            "主屏幕出现 ZOTAIX 图标，点开即全屏运行，支持离线壳与推送式回访。",
          ]
        : [
            "Open this page in Safari (inside WeChat, tap ··· top-right and choose “Open in browser” first).",
            "Tap the Share button in the bottom toolbar (the square with an upward arrow).",
            "Scroll the sheet and choose “Add to Home Screen”.",
            "Confirm the name ZOTAIX and tap “Add”.",
            "The ZOTAIX icon appears on your home screen and opens full-screen with an offline shell.",
          ],
    },
    {
      id: "android-chrome",
      title: zh ? "Android Chrome · 安装应用" : "Android Chrome · Install app",
      badge: "Android",
      steps: zh
        ? [
            "在 Chrome 中打开本页。",
            "点击右上角 ⋮ 菜单，选择「安装应用」或「添加到主屏幕」。",
            "在弹窗中确认安装。",
            "ZOTAIX 会像原生应用一样出现在应用抽屉与主屏幕。",
          ]
        : [
            "Open this page in Chrome.",
            "Tap the ⋮ menu in the top-right corner and choose “Install app” (or “Add to Home screen”).",
            "Confirm the install dialog.",
            "ZOTAIX appears in your app drawer and home screen like a native app.",
          ],
    },
    {
      id: "desktop",
      title: zh ? "桌面 Chrome / Edge · 安装" : "Desktop Chrome / Edge · Install",
      badge: zh ? "桌面端" : "Desktop",
      steps: zh
        ? [
            "在 Chrome 或 Edge 中打开本页。",
            "点击地址栏右侧的安装图标（一个带向下箭头的显示器）。",
            "或者打开浏览器菜单：Chrome 选「投放、保存和分享 → 安装 ZOTAIX」，Edge 选「应用 → 安装此站点为应用」。",
            "安装后 ZOTAIX 以独立窗口运行，可固定到任务栏或 Dock。",
          ]
        : [
            "Open this page in Chrome or Edge.",
            "Click the install icon at the right end of the address bar (a monitor with a downward arrow).",
            "Or use the browser menu: in Chrome choose “Cast, save and share → Install ZOTAIX”; in Edge choose “Apps → Install this site as an app”.",
            "ZOTAIX then runs in its own window and can be pinned to your taskbar or Dock.",
          ],
    },
  ];

  return (
    <div className="space-y-5">
      {isWeChat && (
        <Notice tone="gold" title={zh ? "你正在微信内置浏览器中" : "You are inside WeChat’s in-app browser"}>
          {zh
            ? "微信内置浏览器不提供安装入口。请点击右上角 ··· 菜单，选择「在浏览器中打开」，再回到本页完成安装。"
            : "WeChat’s in-app browser does not expose an install entry. Tap the ··· menu in the top-right corner, choose “Open in system browser”, then return to this page to install."}
        </Notice>
      )}

      <div className="zx-card p-5 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="font-display text-lg text-porcelain">
              {zh ? "在这台设备上安装 ZOTAIX" : "Install ZOTAIX on this device"}
            </p>
            <p className="mt-1 max-w-xl text-sm leading-relaxed text-mist">
              {zh
                ? "网页应用与商店版本共享同一账号、同一档案与同一 AI 礼宾，支持全屏运行与离线壳。"
                : "The web app shares the same account, the same Reserve archive, and the same AI concierge as the store builds — full-screen, with an offline shell."}
            </p>
          </div>
          <div className="shrink-0">
            {isStandalone || installState === "installed" || installState === "accepted" ? (
              <span className="inline-flex items-center gap-2 rounded-md border border-jade/40 bg-jade/10 px-4 py-2.5 text-sm font-medium text-jade">
                ✓ {zh ? "已安装 · 正以应用方式运行" : "Installed · running as an app"}
              </span>
            ) : installEvent ? (
              <Button variant="gold" onClick={install} disabled={installState === "prompting"}>
                {installState === "prompting" ? (zh ? "等待确认…" : "Waiting for confirmation…") : zh ? "一键安装 ZOTAIX" : "Install ZOTAIX"}
              </Button>
            ) : (
              <span className="inline-flex max-w-60 items-center rounded-md border border-hairline px-4 py-2.5 text-xs leading-relaxed text-mist">
                {zh
                  ? "当前浏览器通过自身菜单提供安装入口——按下方对应指南操作即可。"
                  : "This browser offers installation through its own menu — follow the matching guide below."}
              </span>
            )}
          </div>
        </div>
        {installState === "dismissed" && (
          <p className="mt-3 text-xs text-mist">
            {zh
              ? "已取消本次安装。你可以随时再次点击安装，或按下方指南手动添加。"
              : "Install dismissed. You can trigger it again anytime, or add ZOTAIX manually with the guides below."}
          </p>
        )}
      </div>

      <div className="space-y-3">
        {guides.map((g, i) => {
          const open = openGuide === i;
          return (
            <div key={g.id} className="zx-card overflow-hidden !p-0">
              <button
                type="button"
                onClick={() => setOpenGuide(open ? null : i)}
                aria-expanded={open}
                className="flex w-full items-center justify-between gap-3 px-5 py-4 text-left transition-colors hover:bg-veil"
              >
                <span className="flex flex-wrap items-center gap-3">
                  <span className="font-display text-sm text-porcelain sm:text-base">{g.title}</span>
                  <span className="inline-flex items-center rounded-full border border-hairline px-2.5 py-0.5 text-xs text-mist">
                    {g.badge}
                  </span>
                </span>
                <span className={`text-gold transition-transform ${open ? "rotate-90" : ""}`}>→</span>
              </button>
              {open && (
                <ol className="zx-fade-up space-y-2.5 border-t border-hairline px-5 py-4">
                  {g.steps.map((step, n) => (
                    <li key={step} className="flex items-start gap-3 text-sm leading-relaxed text-mist">
                      <span className="font-display shrink-0 text-gold/70">{String(n + 1).padStart(2, "0")}</span>
                      <span>{step}</span>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
