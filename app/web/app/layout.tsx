import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import PageTransition from "../components/layout/PageTransition";
import Sidebar from "../components/layout/Sidebar";
import TitleBar from "../components/layout/TitleBar";

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
      <body className={inter.variable}>
        <TitleBar />
        <div className="flex w-screen h-screen gap-0 overflow-hidden pt-12">
          <Sidebar />
          <main className="flex-1 overflow-auto p-4">
            <PageTransition>{children}</PageTransition>
          </main>
        </div>
      </body>
    </html>
  );
}
