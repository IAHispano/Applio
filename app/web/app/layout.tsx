import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import PageTransition from "../components/layout/PageTransition";
import RoutePrewarm from "../components/layout/RoutePrewarm";
import Sidebar from "../components/layout/Sidebar";
import TermsModal from "../components/setup/TermsModal";
import { I18nProvider } from "../lib/i18n";
import { ThemeProvider } from "../lib/theme";
import Toaster from "../lib/toast";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  weight: ["300", "400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Applio",
  description: "A simple, high-quality voice conversion tool.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body
        className={`${inter.variable} bg-[#0a0a0a] text-neutral-200 overflow-hidden h-screen w-screen flex flex-col m-0 p-0`}
      >
        <I18nProvider>
          <ThemeProvider>
            {/* Accessibility: Skip to Main Content Link for keyboard and screen reader navigation */}
            <a
              href="#main-content"
              className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-[100] focus:px-4 focus:py-2 focus:bg-white focus:text-black focus:font-semibold focus:rounded-lg focus:shadow-xl focus:outline-2 focus:outline-white"
            >
              Skip to main content
            </a>
            <div className="flex flex-1 min-h-0 overflow-hidden">
              <Sidebar />
              <main
                id="main-content"
                tabIndex={-1}
                className="flex-1 min-h-0 min-w-0 overflow-y-auto px-3 sm:px-4 pt-3 pb-3 outline-none"
              >
                <PageTransition>{children}</PageTransition>
              </main>
            </div>
            <TermsModal />
            <Toaster />
            <RoutePrewarm />
          </ThemeProvider>
        </I18nProvider>
      </body>
    </html>
  );
}
