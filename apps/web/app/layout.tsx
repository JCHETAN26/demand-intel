import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "Demand Intelligence",
  description: "SKU-level demand forecasts, uncertainty, drivers, recall signals and replenishment.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <aside className="rail">
            <div className="brand">Demand Intelligence<small>Walmart M5 · CA, TX, WI</small></div>
            <Nav />
          </aside>
          <main className="main">{children}</main>
        </div>
      </body>
    </html>
  );
}
