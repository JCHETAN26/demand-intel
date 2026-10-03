"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  ["/", "Overview"],
  ["/forecast", "Product forecast"],
  ["/drivers", "Forecast drivers"],
  ["/events", "External intelligence"],
  ["/inventory", "Inventory"],
  ["/lab", "Model lab"],
] as const;

export default function Nav() {
  const path = usePathname();
  return (
    <nav className="nav" aria-label="Pages">
      {LINKS.map(([href, label]) => (
        <Link key={href} href={href} aria-current={path === href ? "page" : undefined}>{label}</Link>
      ))}
    </nav>
  );
}
