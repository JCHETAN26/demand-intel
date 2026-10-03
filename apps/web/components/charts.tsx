"use client";
import { useMemo, useRef, useState } from "react";
import { fmt } from "@/lib/api";

const W = 880;

function scaleLinear(d0: number, d1: number, r0: number, r1: number) {
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  return (v: number) => r0 + (v - d0) * k;
}

function niceTicks(max: number, n = 4): number[] {
  if (max <= 0) return [0, 1];
  const step = Math.pow(10, Math.floor(Math.log10(max / n)));
  const m = [1, 2, 2.5, 5, 10].find((f) => (max / (f * step)) <= n) ?? 10;
  const s = m * step;
  const out: number[] = [];
  for (let v = 0; v <= max + 1e-9; v += s) out.push(+v.toFixed(6));
  return out;
}

type Tip = { x: number; y: number; html: React.ReactNode } | null;

function useTip() {
  const [tip, setTip] = useState<Tip>(null);
  const wrap = useRef<HTMLDivElement>(null);
  return { tip, setTip, wrap };
}

/** History (actual units) + forecast mean with P10-P90 and P25-P75 bands; one y-axis. */
export function ForecastChart(props: {
  histDates: string[]; hist: number[];
  fDates: string[]; mean: number[]; p10: number[]; p25: number[]; p75: number[]; p90: number[];
  height?: number;
}) {
  const H = props.height ?? 300;
  const m = { l: 40, r: 12, t: 12, b: 26 };
  const dates = [...props.histDates, ...props.fDates];
  const n = dates.length;
  const ymax = Math.max(1, ...props.hist, ...props.p90) * 1.08;
  const x = scaleLinear(0, n - 1, m.l, W - m.r);
  const y = scaleLinear(0, ymax, H - m.b, m.t);
  const h0 = props.histDates.length;
  const path = (vals: number[], off: number) => vals.map((v, i) => `${i ? "L" : "M"}${x(i + off).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const band = (lo: number[], hi: number[]) =>
    `${path(hi, h0)}${lo.map((v, i) => `L${x(h0 + lo.length - 1 - i).toFixed(1)},${y(lo[lo.length - 1 - i]).toFixed(1)}`).join("")}Z`;
  const ticks = niceTicks(ymax);
  const monthTicks = dates.map((d, i) => [d, i] as const).filter(([d]) => d.endsWith("-01"));
  const { tip, setTip, wrap } = useTip();
  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
    const px = ((e.clientX - box.left) / box.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / (W - m.l - m.r)) * (n - 1))));
    const f = i - h0;
    setTip({
      x: (x(i) / W) * box.width, y: 8,
      html: f < 0
        ? <><b>{dates[i]}</b><br />Sold {props.hist[i]} units</>
        : <><b>{dates[i]}</b> (forecast)<br />Mean {fmt(props.mean[f], 1)} · P50 band {fmt(props.p25[f], 1)}–{fmt(props.p75[f], 1)}<br />P10–P90 {fmt(props.p10[f], 1)}–{fmt(props.p90[f], 1)}</>,
    });
  };
  return (
    <div className="chart" ref={wrap} style={{ position: "relative" }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Daily units sold and forecast with uncertainty bands">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} stroke="var(--grid)" strokeWidth={1} />
            <text x={m.l - 6} y={y(t) + 4} textAnchor="end">{t}</text>
          </g>
        ))}
        {monthTicks.map(([d, i]) => <text key={d} x={x(i)} y={H - 8} textAnchor="middle">{d.slice(0, 7)}</text>)}
        <path d={band(props.p10, props.p90)} fill="var(--band-outer)" stroke="none" />
        <path d={band(props.p25, props.p75)} fill="var(--band-inner)" stroke="none" opacity={0.75} />
        <path d={path(props.hist, 0)} fill="none" stroke="var(--ink-2)" strokeWidth={1.5} />
        <path d={path(props.mean, h0)} fill="none" stroke="var(--series-1)" strokeWidth={2} />
        <line x1={x(h0 - 0.5)} x2={x(h0 - 0.5)} y1={m.t} y2={H - m.b} stroke="var(--axis)" strokeDasharray="3 3" />
        <text x={x(h0 - 0.5) + 4} y={m.t + 10}>forecast origin</text>
        <line x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} stroke="var(--axis)" />
        <rect x={m.l} y={m.t} width={W - m.l - m.r} height={H - m.t - m.b} fill="transparent"
          onMouseMove={onMove} onMouseLeave={() => setTip(null)} />
      </svg>
      {tip && <div className="tip" style={{ left: Math.min(tip.x + 10, (wrap.current?.clientWidth ?? 600) - 220), top: tip.y }}>{tip.html}</div>}
      <div className="legend">
        <span><i style={{ background: "var(--ink-2)", height: 2 }} />Units sold</span>
        <span><i style={{ background: "var(--series-1)", height: 3 }} />Forecast mean</span>
        <span><i style={{ background: "var(--band-inner)" }} />P25–P75</span>
        <span><i style={{ background: "var(--band-outer)" }} />P10–P90</span>
      </div>
    </div>
  );
}

/** Sell price on its own panel (never a second axis on the demand chart). */
export function PriceChart({ dates, price, height = 110 }: { dates: string[]; price: (number | null)[]; height?: number }) {
  const m = { l: 40, r: 12, t: 10, b: 20 };
  const vals = price.filter((p): p is number => p !== null);
  if (!vals.length) return <p className="muted small">No price in this window (item not on sale).</p>;
  const lo = Math.min(...vals) * 0.95, hi = Math.max(...vals) * 1.05;
  const x = scaleLinear(0, dates.length - 1, m.l, W - m.r);
  const y = scaleLinear(lo, hi, height - m.b, m.t);
  let d = "";
  price.forEach((p, i) => { if (p !== null) d += `${d && price[i - 1] !== null ? "L" : "M"}${x(i).toFixed(1)},${y(p).toFixed(1)}`; });
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label="Sell price over time">
        {[lo, hi].map((t) => <text key={t} x={m.l - 6} y={y(t) + 4} textAnchor="end">${t.toFixed(2)}</text>)}
        <path d={d} fill="none" stroke="var(--series-2)" strokeWidth={2} />
        <line x1={m.l} x2={W - m.r} y1={height - m.b} y2={height - m.b} stroke="var(--axis)" />
      </svg>
    </div>
  );
}

/** Diverging horizontal bars: contribution of each feature family in units. */
export function DriverBars({ values }: { values: Record<string, number> }) {
  const rows = useMemo(() => Object.entries(values).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])), [values]);
  const max = Math.max(1e-6, ...rows.map(([, v]) => Math.abs(v)));
  const rowH = 28, labelW = 190, H = rows.length * rowH + 8;
  const mid = labelW + (W - labelW - 70) / 2;
  const sx = (v: number) => (v / max) * ((W - labelW - 70) / 2);
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Forecast drivers in units">
        <line x1={mid} x2={mid} y1={0} y2={H} stroke="var(--axis)" />
        {rows.map(([k, v], i) => {
          const w = Math.abs(sx(v));
          const xx = v >= 0 ? mid : mid - w;
          return (
            <g key={k} transform={`translate(0, ${i * rowH + 4})`}>
              <text x={labelW - 8} y={16} textAnchor="end" style={{ fill: "var(--ink-2)" }}>{k}</text>
              <rect x={xx} y={4} width={Math.max(w, 1)} height={16} rx={3} fill={v >= 0 ? "var(--pos)" : "var(--neg)"}>
                <title>{`${k}: ${v >= 0 ? "+" : ""}${v.toFixed(2)} units`}</title>
              </rect>
              <text x={v >= 0 ? mid + w + 6 : mid - w - 6} y={16} textAnchor={v >= 0 ? "start" : "end"} style={{ fill: "var(--ink)" }}>
                {v >= 0 ? "+" : ""}{v.toFixed(1)}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="legend">
        <span><i style={{ background: "var(--pos)" }} />Raises the forecast</span>
        <span><i style={{ background: "var(--neg)" }} />Lowers the forecast</span>
      </div>
    </div>
  );
}

/** Horizontal bars for one metric across models (lower is better), with a reference line. */
export function CompareBars({ rows, metric, reference }: { rows: { label: string; value: number }[]; metric: string; reference?: number }) {
  const sorted = [...rows].filter((r) => Number.isFinite(r.value)).sort((a, b) => a.value - b.value);
  const max = Math.max(...sorted.map((r) => r.value), reference ?? 0) * 1.1;
  const rowH = 24, labelW = 190, H = sorted.length * rowH + 8;
  const sx = scaleLinear(0, max, labelW, W - 60);
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${metric} by model`}>
        {sorted.map((r, i) => (
          <g key={r.label} transform={`translate(0, ${i * rowH + 4})`}>
            <text x={labelW - 8} y={14} textAnchor="end" style={{ fill: "var(--ink-2)" }}>{r.label}</text>
            <rect x={labelW} y={3} width={Math.max(sx(r.value) - labelW, 1)} height={14} rx={3} fill="var(--series-1)">
              <title>{`${r.label}: ${r.value.toFixed(4)}`}</title>
            </rect>
            <text x={sx(r.value) + 6} y={14} style={{ fill: "var(--ink)" }}>{r.value.toFixed(3)}</text>
          </g>
        ))}
        {reference !== undefined && (
          <line x1={sx(reference)} x2={sx(reference)} y1={0} y2={H} stroke="var(--critical)" strokeDasharray="4 3">
            <title>{`seasonal naive: ${reference.toFixed(4)}`}</title>
          </line>
        )}
      </svg>
    </div>
  );
}
