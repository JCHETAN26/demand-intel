"use client";
import { useEffect, useState } from "react";
import SeriesPicker, { type Pick } from "@/components/SeriesPicker";
import { fmt, pct, post, type Recommendation } from "@/lib/api";

export default function Inventory() {
  const [pick, setPick] = useState<Pick | null>(null);
  const [onHand, setOnHand] = useState(20);
  const [onOrder, setOnOrder] = useState(0);
  const [lead, setLead] = useState(7);
  const [sl, setSl] = useState(0.95);
  const [rec, setRec] = useState<Recommendation | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    if (!pick) return;
    const t = setTimeout(() => {
      post<Recommendation>("/v1/inventory/recommend", { ...pick, current_inventory: onHand, on_order: onOrder, lead_time_days: lead, service_level: sl })
        .then((r) => { setRec(r); setErr(null); }).catch((e: Error) => setErr(e.message));
    }, 250);
    return () => clearTimeout(t);
  }, [pick, onHand, onOrder, lead, sl]);
  return (
    <>
      <header>
        <h1>Inventory</h1>
        <p className="lede">Order-up-to recommendation from the forecast distribution. Lead-time demand covers the lead time plus one review day, simulated from the daily quantiles with a day-to-day correlation calibrated on backtests. Change the assumptions to see the effect.</p>
      </header>
      <SeriesPicker onChange={setPick} />
      <section className="row">
        <label className="f" htmlFor="inv-onhand">Current stock (units)<input id="inv-onhand" type="number" min={0} value={onHand} onChange={(e) => setOnHand(Math.max(0, +e.target.value))} /></label>
        <label className="f" htmlFor="inv-onorder">On order<input id="inv-onorder" type="number" min={0} value={onOrder} onChange={(e) => setOnOrder(Math.max(0, +e.target.value))} /></label>
        <label className="f" htmlFor="inv-lead">Lead time (days)<input id="inv-lead" type="number" min={1} max={21} value={lead} onChange={(e) => setLead(Math.min(21, Math.max(1, +e.target.value)))} /></label>
        <label className="f" htmlFor="inv-sl">Target service level<select id="inv-sl" value={sl} onChange={(e) => setSl(+e.target.value)}>
          {[0.9, 0.95, 0.98].map((v) => <option key={v} value={v}>{pct(v, 0)}</option>)}</select></label>
      </section>
      {err && <p className="error">{err}</p>}
      {rec && (
        <section className="tiles">
          <div className="tile"><span className="k">Recommended order</span><span className="v">{rec.recommended_order} units</span><span className="s">to reach the order-up-to level</span></div>
          <div className="tile"><span className="k">Lead-time demand</span><span className="v">{fmt(rec.p50, 0)} – {fmt(rec.p90, 0)}</span><span className="s">P50 – P90 over {rec.lead_time_days + 1} days</span></div>
          <div className="tile"><span className="k">Order-up-to level</span><span className="v">{fmt(rec.target_level, 0)}</span><span className="s">{pct(rec.service_level, 0)} quantile</span></div>
          <div className="tile"><span className="k">Safety stock</span><span className="v">{fmt(rec.safety_stock, 1)}</span><span className="s">above expected {fmt(rec.expected_demand, 1)}</span></div>
          <div className="tile"><span className="k">Stockout chance now</span><span className="v">{pct(rec.stockout_probability, 0)}</span><span className="s">if nothing is ordered</span></div>
          <div className="tile"><span className="k">Inventory position</span><span className="v">{fmt(rec.inventory_position, 0)}</span><span className="s">on hand + on order</span></div>
        </section>
      )}
      {rec && <p className="muted small">Day-to-day correlation ρ = {rec.rho}. {rec.note} Stock levels are your inputs: M5 contains no inventory data.</p>}
    </>
  );
}
