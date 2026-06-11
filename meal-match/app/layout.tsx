import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";
import { StoreProvider } from "@/lib/store";

export const metadata: Metadata = {
  title: "Meal Match — find a cook, find your table",
  description:
    "A home-cook marketplace that matches you with a cook and the meal buddies to share the table.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <StoreProvider>
          <header className="site-header">
            <div className="container bar">
              <Link href="/" className="brand">
                Meal<span className="dot">Match</span>
              </Link>
              <nav className="nav">
                <Link href="/">Browse</Link>
                <Link href="/match">Find my table</Link>
                <Link href="/tables">My tables</Link>
              </nav>
            </div>
          </header>
          <main className="container">{children}</main>
          <footer className="footer">
            <div className="container">
              MealMatch · a Deliveroo alternative built on home cooks and shared
              tables. Demo data, no real orders.
            </div>
          </footer>
        </StoreProvider>
      </body>
    </html>
  );
}
