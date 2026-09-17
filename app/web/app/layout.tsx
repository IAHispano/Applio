import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import PageTransition from "../components/layout/PageTransition";
import Sidebar from "../components/layout/Sidebar";
import TitleBar from "../components/layout/TitleBar";
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
            <TitleBar />
            <div className="flex flex-1 min-h-0 overflow-hidden">
              <Sidebar />
              <main className="flex-1 min-h-0 min-w-0 overflow-y-auto p-4">
                <PageTransition>{children}</PageTransition>
              </main>
            </div>
            <Toaster />
          </ThemeProvider>
        </I18nProvider>
      </body>
    </html>
  );
}
