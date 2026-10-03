"use client";
import { Suspense, useState } from "react";
import { useSearchParams } from "next/navigation";
import SeriesPicker, { type Pick } from "@/components/SeriesPicker";
import { ForecastChart, PriceChart } from "@/components/charts";
import { useApi } from "@/components/useApi";
import { fmt, type SeriesDetail } from "@/lib/api";

function Page() {
  const params = useSearchParams();
  const initial = params.get("item") && params.get("store") ? { item_id: params.get("item")!, store_id: params.get("store")! } : null;
  const [pick, setPick] = useState<Pick | null>(initial);
  const [showTable, setShowTable] = useState(false);
  const { data, error } = useApi<SeriesDetail>(pick ? `/v1/series/${pick.item_id}/${pick.store_id}` : null);
  const hist = data ? { d: data.history.dates.slice(-91), s: data.history.sales.slice(-91), p: data.history.price.slice(-91) } : null;
  return (
    <>
      <header>
        <h1>Product forecast</h1>
        <p className="lede">The last 13 weeks of sales and the next 28 days of forecast. Bands show where demand is expected to fall: half the time inside the dark band, 80% of the time inside the light one.</p>
      </header>
      <SeriesPicker onChange={(p) => { if (!initial || p.item_id !== initial.item_id) setPick(p); }} />
      {error && <p className="error">{error}</p>}
      {data && hist && (
        <>
          <section className="panel">
            <div className="row"><h2>{data.series.item_id} at {data.series.store_id}</h2>
              <span className="pill">{data.series.demand_class}</span><span className="pill">{data.series.volume_tercile} volume</span></div>
            <ForecastChart histDates={hist.d} hist={hist.s} fDates={data.forecast.dates} mean={data.forecast.mean}
              p10={data.forecast.p10} p25={data.forecast.p25} p75={data.forecast.p75} p90={data.forecast.p90} />
          </section>
          <section className="panel">
            <h2>Sell price</h2>
            <PriceChart dates={[...hist.d, ...data.forecast.dates]} price={[...hist.p, ...data.forecast.price]} />
            <p className="muted small">Discounts are inferred from price drops: M5 has no promotion flag.</p>
          </section>
          <section className="panel">
            <div className="row"><h2>Forecast table</h2><button className="ghost" onClick={() => setShowTable(!showTable)}>{showTable ? "Hide" : "Show"} table</button></div>
            {showTable && (
              <div className="tablewrap"><table>
                <thead><tr><th>Date</th><th className="num">Mean</th><th className="num">P10</th><th className="num">P50</th><th className="num">P90</th><th className="num">Price</th></tr></thead>
                <tbody>{data.forecast.dates.map((d, i) => (
                  <tr key={d}><td>{d}</td><td className="num">{fmt(data.forecast.mean[i], 2)}</td><td className="num">{fmt(data.forecast.p10[i], 1)}</td>
                    <td className="num">{fmt(data.forecast.p50[i], 1)}</td><td className="num">{fmt(data.forecast.p90[i], 1)}</td>
                    <td className="num">{data.forecast.price[i] === null ? "—" : `$${fmt(data.forecast.price[i], 2)}`}</td></tr>))}
                </tbody></table></div>
            )}
          </section>
        </>
      )}
    </>
  );
}

export default function ForecastPage() {
  return <Suspense fallback={<p className="muted">Loading…</p>}><Page /></Suspense>;
}
