import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Dormant Viewer",
  description: "Batch viewer for JS Dormant puzzle probing",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        {/* Top bar */}
        <header className="h-9 border-b flex items-center px-4 gap-6" style={{ borderColor: "var(--border)", background: "var(--surface)" }}>
          <Link href="/" className="font-bold text-sm tracking-tight" style={{ color: "var(--foreground)" }}>
            dormant
          </Link>
          <nav className="flex gap-4">
            <Link href="/batches" className="text-xs hover:underline" style={{ color: "var(--muted)" }}>
              batches
            </Link>
            <Link href="/chat" className="text-xs hover:underline" style={{ color: "var(--muted)" }}>
              chat
            </Link>
          </nav>
        </header>

        <main>{children}</main>
      </body>
    </html>
  );
}
