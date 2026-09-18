"use client";

import {
  AlertTriangle,
  ArrowDown,
  Ban,
  CheckCircle2,
  Database,
  ExternalLink,
  FileClock,
  FileText,
  Shield,
  ShieldAlert,
  Sparkles,
  UserCheck,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useI18n } from "../../lib/i18n";

const TERMS_STORAGE_KEY = "applio:terms-accepted";
const TERMS_DATE_KEY = "applio:terms-accepted-at";

export function hasAcceptedTerms(): boolean {
  if (typeof window === "undefined") return true;
  try {
    return localStorage.getItem(TERMS_STORAGE_KEY) === "true";
  } catch {
    return true;
  }
}

export default function TermsModal() {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const [scrolledToBottom, setScrolledToBottom] = useState(false);
  const [scrollProgress, setScrollProgress] = useState(0);
  const [agreed, setAgreed] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Only show if terms haven't been accepted yet
    if (!hasAcceptedTerms()) {
      setOpen(true);
    }
  }, []);

  const handleScroll = () => {
    const el = scrollRef.current;
    if (!el) return;

    const maxScroll = el.scrollHeight - el.clientHeight;
    if (maxScroll <= 0) {
      setScrollProgress(100);
      setScrolledToBottom(true);
      return;
    }

    const current = Math.min(maxScroll, Math.max(0, el.scrollTop));
    const progress = Math.round((current / maxScroll) * 100);
    setScrollProgress(progress);

    // Considered read if scrolled within 40px of bottom or >= 98%
    const isAtBottom = maxScroll - current < 40 || progress >= 98;
    if (isAtBottom && !scrolledToBottom) {
      setScrolledToBottom(true);
    }
  };

  const scrollToBottom = () => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  };

  const handleAccept = () => {
    try {
      localStorage.setItem(TERMS_STORAGE_KEY, "true");
      localStorage.setItem(TERMS_DATE_KEY, new Date().toISOString());
    } catch {
      /* non-fatal */
    }
    setOpen(false);
  };

  if (!open) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="terms-modal-title"
      className="fixed inset-0 z-[9999] flex items-center justify-center p-3 sm:p-5 bg-black/85 backdrop-blur-xl animate-in fade-in duration-200"
    >
      <div className="relative w-full max-w-3xl max-h-[92vh] flex flex-col rounded-2xl border border-white/15 bg-[#121212]/95 backdrop-blur-2xl shadow-2xl overflow-hidden">
        {/* Top ambient highlight line */}
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/30 to-transparent pointer-events-none" />

        {/* Modal Header */}
        <div className="p-5 sm:p-6 border-b border-white/10 bg-white/[0.02] flex items-center justify-between gap-4">
          <div className="flex items-center gap-3.5 min-w-0">
            <div className="w-11 h-11 rounded-xl bg-white/[0.08] border border-white/15 flex items-center justify-center text-white shrink-0 shadow-inner">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <div className="space-y-0.5 min-w-0">
              <div className="flex items-center gap-2">
                <h2 id="terms-modal-title" className="text-lg font-bold text-white tracking-tight m-0">
                  {t("Terms of Use")}
                </h2>
                <span className="text-[10px] uppercase px-2 py-0.5 rounded-full bg-white/10 border border-white/15 text-neutral-300">
                  Applio
                </span>
              </div>
              <p className="text-xs text-neutral-400 m-0 truncate">
                {t("Please review and accept our official terms before using Applio.")}
              </p>
            </div>
          </div>

          {/* Reading progress indicator in header */}
          <div className="hidden sm:flex flex-col items-end gap-1 shrink-0 select-none">
            <span className="text-[11px] text-neutral-400">
              {scrolledToBottom ? (
                <span className="text-emerald-400 font-semibold flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  {t("Document Read")}
                </span>
              ) : (
                <span>
                  {scrollProgress}% {t("read")}
                </span>
              )}
            </span>
            <div className="w-24 h-1.5 rounded-full bg-white/10 overflow-hidden">
              <div
                className={`h-full transition-all duration-150 ${
                  scrolledToBottom ? "bg-emerald-400" : "bg-white"
                }`}
                style={{ width: `${scrollProgress}%` }}
              />
            </div>
          </div>
        </div>

        {/* Slim reading progress bar for mobile / full width */}
        <div className="sm:hidden w-full h-1 bg-white/10 overflow-hidden">
          <div
            className={`h-full transition-all duration-150 ${
              scrolledToBottom ? "bg-emerald-400" : "bg-white"
            }`}
            style={{ width: `${scrollProgress}%` }}
          />
        </div>

        {/* Scrollable Terms Content - Verbatim from TERMS_OF_USE.md */}
        <div className="relative flex-1 overflow-hidden flex flex-col min-h-0">
          <section
            ref={scrollRef}
            onScroll={handleScroll}
            aria-label={t("Terms of Use document")}
            className="flex-1 overflow-y-auto p-5 sm:p-7 space-y-6 text-xs sm:text-sm text-neutral-300 leading-relaxed font-sans focus:outline-none scrollbar-thin"
          >
            {/* Top Important Banner */}
            <div className="p-4 rounded-xl border border-amber-500/25 bg-amber-500/[0.08] text-amber-200 flex items-start gap-3.5 shadow-sm">
              <ShieldAlert className="w-5 h-5 shrink-0 mt-0.5 text-amber-400" />
              <div className="space-y-0.5">
                <h3 className="text-xs font-semibold text-amber-300 uppercase tracking-wider m-0">
                  {t("Important Notice")}
                </h3>
                <p className="text-xs text-amber-200/90 m-0 leading-normal">
                  {t(
                    "Applio is an open-source audio research and voice transformation tool. Responsible, ethical, and legal use is strictly required.",
                  )}
                </p>
              </div>
            </div>

            {/* Section 1: Responsibilities of the User */}
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4 sm:p-5 space-y-3.5 hover:border-white/20 transition-colors">
              <div className="flex items-center gap-2 text-white">
                <div className="w-6 h-6 rounded-lg bg-white/10 flex items-center justify-center text-white shrink-0">
                  <UserCheck className="w-3.5 h-3.5" />
                </div>
                <h3 className="text-sm font-bold text-white tracking-tight m-0">
                  {t("Responsibilities of the User")}
                </h3>
              </div>
              <p className="text-xs text-neutral-400 m-0">
                {t("By using Applio, you agree to the following responsibilities:")}
              </p>

              <div className="space-y-3 pt-1">
                {/* 1. Respect Intellectual Property */}
                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-1.5">
                  <h4 className="text-xs font-semibold text-neutral-200 m-0">
                    1. {t("Respect Intellectual Property and Privacy Rights")}
                  </h4>
                  <ul className="list-disc pl-4 space-y-1 text-xs text-neutral-400">
                    <li>
                      {t(
                        "Ensure that any audio or material processed through Applio is either owned by you or used with explicit permission from the rightful owner.",
                      )}
                    </li>
                    <li>
                      {t(
                        "Respect copyrights, intellectual property rights, and privacy rights of all individuals and entities.",
                      )}
                    </li>
                  </ul>
                </div>

                {/* 2. Avoid Harmful or Unethical Use */}
                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-1.5">
                  <h4 className="text-xs font-semibold text-neutral-200 m-0">
                    2. {t("Avoid Harmful or Unethical Use")}
                  </h4>
                  <ul className="list-disc pl-4 space-y-1 text-xs text-neutral-400">
                    <li>
                      {t(
                        "Do not use Applio to create or distribute content that harms, defames, or infringes upon the rights of others.",
                      )}
                    </li>
                    <li>
                      {t(
                        "Avoid any activities that may violate ethical standards, promote hate speech, or facilitate illegal conduct.",
                      )}
                    </li>
                  </ul>
                </div>

                {/* 3. Adhere to Local Laws */}
                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-1.5">
                  <h4 className="text-xs font-semibold text-neutral-200 m-0">
                    3. {t("Adhere to Local Laws and Regulations")}
                  </h4>
                  <p className="text-xs text-neutral-400 m-0 leading-relaxed">
                    {t(
                      "Familiarize yourself with and comply with the laws and regulations governing the use of AI, voice transformation tools, and generated content in your jurisdiction.",
                    )}
                  </p>
                </div>
              </div>
            </div>

            {/* Section 2: Disclaimer of Liability */}
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4 sm:p-5 space-y-3.5 hover:border-white/20 transition-colors">
              <div className="flex items-center gap-2 text-white">
                <div className="w-6 h-6 rounded-lg bg-white/10 flex items-center justify-center text-white shrink-0">
                  <Shield className="w-3.5 h-3.5" />
                </div>
                <h3 className="text-sm font-bold text-white tracking-tight m-0">
                  {t("Disclaimer of Liability")}
                </h3>
              </div>
              <p className="text-xs text-neutral-400 m-0">
                {t(
                  "Applio and its contributors disclaim all liability for any misuse or unintended consequences arising from the use of this tool.",
                )}
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 pt-1">
                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-1">
                  <strong className="text-xs font-semibold text-white block">{t("No Warranty")}</strong>
                  <p className="text-[11px] text-neutral-400 m-0 leading-normal">
                    {t('Applio is provided "as is" without any warranty, express or implied.')}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-1">
                  <strong className="text-xs font-semibold text-white block">
                    {t("User Responsibility")}
                  </strong>
                  <p className="text-[11px] text-neutral-400 m-0 leading-normal">
                    {t(
                      "You bear full responsibility for how you choose to use Applio and any outcomes resulting from that use.",
                    )}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-1">
                  <strong className="text-xs font-semibold text-white block">{t("No Endorsement")}</strong>
                  <p className="text-[11px] text-neutral-400 m-0 leading-normal">
                    {t(
                      "Applio does not endorse or support any activities or content created with this tool that result in harm, illegal activity, or unethical practices.",
                    )}
                  </p>
                </div>
              </div>
            </div>

            {/* Section 3: Permitted Use Cases */}
            <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/[0.02] p-4 sm:p-5 space-y-3.5 hover:border-emerald-500/30 transition-colors">
              <div className="flex items-center gap-2 text-emerald-300">
                <div className="w-6 h-6 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 shrink-0">
                  <Sparkles className="w-3.5 h-3.5" />
                </div>
                <h3 className="text-sm font-bold text-white tracking-tight m-0">
                  {t("Permitted Use Cases")}
                </h3>
              </div>
              <p className="text-xs text-neutral-400 m-0">{t("Applio is designed for:")}</p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-1">
                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-1">
                  <strong className="text-xs font-semibold text-emerald-300 block">
                    {t("Personal Projects")}
                  </strong>
                  <p className="text-[11px] text-neutral-400 m-0 leading-normal">
                    {t("Experimentation and creative endeavors for personal enrichment.")}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-1">
                  <strong className="text-xs font-semibold text-emerald-300 block">
                    {t("Academic Research")}
                  </strong>
                  <p className="text-[11px] text-neutral-400 m-0 leading-normal">
                    {t("Advancing scientific understanding and education.")}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-1">
                  <strong className="text-xs font-semibold text-emerald-300 block">
                    {t("Investigative Purposes")}
                  </strong>
                  <p className="text-[11px] text-neutral-400 m-0 leading-normal">
                    {t("Analyzing data in lawful and ethical contexts.")}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-black/40 border border-white/5 space-y-1">
                  <strong className="text-xs font-semibold text-emerald-300 block">
                    {t("Commercial Use")}
                  </strong>
                  <p className="text-[11px] text-neutral-400 m-0 leading-normal">
                    {t(
                      "Creating content for commercial purposes, provided that appropriate rights and permissions are obtained and all legal and ethical standards are adhered to.",
                    )}
                  </p>
                </div>
              </div>
            </div>

            {/* Section 4: Prohibited Activities */}
            <div className="rounded-xl border border-red-500/25 bg-red-500/[0.03] p-4 sm:p-5 space-y-3.5 hover:border-red-500/40 transition-colors">
              <div className="flex items-center gap-2 text-red-300">
                <div className="w-6 h-6 rounded-lg bg-red-500/10 border border-red-500/20 flex items-center justify-center text-red-400 shrink-0">
                  <Ban className="w-3.5 h-3.5" />
                </div>
                <h3 className="text-sm font-bold text-white tracking-tight m-0">
                  {t("Prohibited Activities")}
                </h3>
              </div>
              <p className="text-xs text-neutral-400 m-0">
                {t("The following uses are explicitly prohibited:")}
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 pt-1">
                <div className="p-3 rounded-lg bg-black/40 border border-red-500/15 space-y-1">
                  <strong className="text-xs font-semibold text-red-300 block">
                    {t("Harmful Applications")}
                  </strong>
                  <p className="text-[11px] text-neutral-400 m-0 leading-normal">
                    {t("Generating audio to defame, harm, or manipulate others.")}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-black/40 border border-red-500/15 space-y-1">
                  <strong className="text-xs font-semibold text-red-300 block">
                    {t("Unauthorized Distribution")}
                  </strong>
                  <p className="text-[11px] text-neutral-400 m-0 leading-normal">
                    {t("Sharing content that violates copyrights or the rights of others.")}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-black/40 border border-red-500/15 space-y-1">
                  <strong className="text-xs font-semibold text-red-300 block">
                    {t("Deceptive Practices")}
                  </strong>
                  <p className="text-[11px] text-neutral-400 m-0 leading-normal">
                    {t("Creating content intended to deceive or defraud others.")}
                  </p>
                </div>
              </div>
            </div>

            {/* Section 5: Training Data */}
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4 sm:p-5 space-y-2 hover:border-white/20 transition-colors">
              <div className="flex items-center gap-2 text-white">
                <div className="w-6 h-6 rounded-lg bg-white/10 flex items-center justify-center text-white shrink-0">
                  <Database className="w-3.5 h-3.5" />
                </div>
                <h3 className="text-sm font-bold text-white tracking-tight m-0">{t("Training Data")}</h3>
              </div>
              <p className="text-xs text-neutral-300 m-0 leading-relaxed pl-1">
                {t(
                  "All official models distributed by Applio have been trained under publicly available datasets such as",
                )}{" "}
                <a
                  href="https://huggingface.co/datasets/IAHispano/Applio-Dataset"
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 text-white font-medium underline underline-offset-2 hover:text-neutral-300"
                >
                  <span>VCTK</span>
                  <ExternalLink className="w-3 h-3 inline" />
                </a>
                .{" "}
                {t(
                  "We strive to maintain transparency and ethical practices in the development and distribution of our tools.",
                )}
              </p>
            </div>

            {/* Section 6: Amendments */}
            <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4 sm:p-5 space-y-2 hover:border-white/20 transition-colors">
              <div className="flex items-center gap-2 text-white">
                <div className="w-6 h-6 rounded-lg bg-white/10 flex items-center justify-center text-white shrink-0">
                  <FileClock className="w-3.5 h-3.5" />
                </div>
                <h3 className="text-sm font-bold text-white tracking-tight m-0">{t("Amendments")}</h3>
              </div>
              <p className="text-xs text-neutral-300 m-0 leading-relaxed pl-1">
                {t(
                  "Applio reserves the right to modify these terms at any time. Continued use of the tool signifies your acceptance of any updated terms.",
                )}
              </p>
            </div>

            <div className="pt-2 text-[11px] text-neutral-500 text-center italic">
              {t("End of document. Continued use of Applio constitutes full acceptance of these terms.")}
            </div>
          </section>

          {/* Quick jump to bottom floating button when not at bottom */}
          {!scrolledToBottom && (
            <button
              type="button"
              onClick={scrollToBottom}
              aria-label={t("Scroll to bottom")}
              className="absolute bottom-4 right-5 py-1.5 px-3 rounded-full bg-neutral-900/90 border border-white/20 text-neutral-200 hover:text-white hover:bg-neutral-800 shadow-xl backdrop-blur flex items-center gap-1.5 text-xs transition-colors"
              style={{ transform: "none" }}
            >
              <span>{t("Scroll down")}</span>
              <ArrowDown className="w-3.5 h-3.5 text-neutral-400 animate-bounce" />
            </button>
          )}
        </div>

        {/* Modal Footer with Agreement Controls */}
        <div className="p-4 sm:p-5 border-t border-white/10 bg-black/50 backdrop-blur-xl flex flex-col sm:flex-row sm:items-center justify-between gap-3 sm:gap-4">
          <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3 min-w-0">
            <label
              className={`flex items-center gap-3 select-none text-xs transition-opacity ${
                !scrolledToBottom ? "opacity-50 cursor-not-allowed" : "cursor-pointer text-neutral-100"
              }`}
            >
              <input
                type="checkbox"
                checked={agreed}
                disabled={!scrolledToBottom}
                onChange={(e) => setAgreed(e.target.checked)}
                className="w-4 h-4 rounded border-white/20 accent-white bg-white/5 disabled:cursor-not-allowed cursor-pointer"
              />
              <span className="font-medium">{t("I have read and agree to the Applio Terms of Use")}</span>
            </label>

            {!scrolledToBottom ? (
              <span className="text-[11px] font-medium text-amber-400/90 flex items-center gap-1">
                <AlertTriangle className="w-3 h-3 shrink-0" />
                {t("Scroll to the bottom to unlock")}
              </span>
            ) : (
              <span className="text-[11px] font-medium text-emerald-400 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3 shrink-0" />
                {t("Unlocked")}
              </span>
            )}
          </div>

          <button
            type="button"
            className="cta flex items-center justify-center gap-2 py-2 px-6 text-xs font-semibold disabled:opacity-30 disabled:pointer-events-none shrink-0"
            disabled={!agreed || !scrolledToBottom}
            onClick={handleAccept}
            style={{ transform: "none" }}
          >
            <CheckCircle2 className="w-4 h-4" />
            <span>{t("Accept & Continue")}</span>
          </button>
        </div>
      </div>
    </div>
  );
}
