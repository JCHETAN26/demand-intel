"use client";
import Link from "next/link";
import { useApi } from "@/components/useApi";
import { fmt, pct, type RiskRow, type Summary } from "@/lib/api";

function RiskTable({ kind }: { kind: "stockout" | "excess" }) {
  const { data, error } = useApi<{ items: RiskRow[] }>(`/v1/risk?kind=${kind}&limit=8`);
  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading…</p>;
  return (
    <div className="tablewrap">
      <table>
        <thead><tr><th>SKU</th><th>Store</th><th className="num">Stock</th><th className="num">8-day P95 need</th></tr></thead>
        <tbody>
          {data.items.map((r) => (
            <tr key={`${r.item_id}-${r.store_id}`}>
              <td><Link href={`/forecast?item=${r.item_id}&store=${r.store_id}`}>{r.item_id}</Link></td>
              <td>{r.store_id}</td>
              <td className="num">{fmt(r.scenario_stock, 0)}</td>
              <td className="num">{fmt(r.lead_time_p95, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Overview() {
  const { data: s, error } = useApi<Summary>("/v1/summary");
  return (
    <>
      <header>
        <h1>Demand Intelligence</h1>
        <p className="lede">
          {s ? <>Forecasts for {s.forecast_dates[0]} to {s.forecast_dates[1]}, made with data through {s.origin_date} by <span className="mono">{s.model}</span>.</> : "Loading forecasts…"}
        </p>
      </header>
      {error && <p className="error">The API is not reachable ({error}). Start it with <span className="mono">make up</span>.</p>}
      {s && (
        <section className="tiles" aria-label="Headline figures">
          <div className="tile"><span className="k">Forecast horizon</span><span className="v">{s.horizon_days} days</span></div>
          <div className="tile"><span className="k">SKUs monitored</span><span className="v">{s.skus_monitored.toLocaleString()}</span><span className="s">item × store series</span></div>
          <div className="tile"><span className="k">Stockout risk</span><span className="v">{s.stockout_risk.toLocaleString()}</span><span className="s">&gt;10% stockout chance under a simple reorder rule</span></div>
          <div className="tile"><span className="k">Excess inventory risk</span><span className="v">{s.excess_inventory_risk.toLocaleString()}</span><span className="s">simple-rule stock over 1.5× the P95 need</span></div>
          <div className="tile"><span className="k">Forecast WAPE</span><span className="v">{pct(s.evaluation.test_wape)}</span><span className="s">test window, 28 days</span></div>
          <div className="tile"><span className="k">Forecast bias</span><span className="v">{pct(s.evaluation.test_bias)}</span><span className="s">negative = under-forecast</span></div>
          <div className="tile"><span className="k">WRMSSE</span><span className="v">{fmt(s.evaluation.test_wrmsse, 3)}</span><span className="s">M5 metric, test window</span></div>
        </section>
      )}
      {s && <p className="muted small">{s.risk_definition} Store inventory is not part of M5, so the stock level is a scenario, not observed data.</p>}
      <section className="grid2">
        <div className="panel"><h2>Highest stockout risk</h2><RiskTable kind="stockout" /></div>
        <div className="panel"><h2>Highest excess-inventory risk</h2><RiskTable kind="excess" /></div>
      </section>
    </>
  );
}
