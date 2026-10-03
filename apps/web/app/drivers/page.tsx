"use client";
import { useMemo, useState } from "react";
import SeriesPicker, { type Pick } from "@/components/SeriesPicker";
import { DriverBars } from "@/components/charts";
import { useApi } from "@/components/useApi";
import { fmt, type SeriesDetail } from "@/lib/api";

const WINDOWS = [["Next 7 days", 0, 7], ["Days 8–14", 7, 14], ["Days 15–28", 14, 28], ["All 28 days", 0, 28]] as const;

export default function Drivers() {
  const [pick, setPick] = useState<Pick | null>(null);
  const [w, setW] = useState(0);
  const { data, error } = useApi<SeriesDetail>(pick ? `/v1/series/${pick.item_id}/${pick.store_id}` : null);
  const values = useMemo(() => {
    if (!data) return null;
    const [, a, b] = WINDOWS[w];
    const out: Record<string, number> = {};
    data.drivers.families.forEach((f, j) => {
      out[f] = data.drivers.daily_units.slice(a, b).reduce((s, row) => s + row[j], 0);
    });
    return out;
  }, [data, w]);
  const total = data ? data.forecast.mean.slice(WINDOWS[w][1], WINDOWS[w][2]).reduce((a, b) => a + b, 0) : 0;
  return (
    <>
      <header>
        <h1>Forecast drivers</h1>
        <p className="lede">How each group of inputs moves this SKU&apos;s forecast away from the model&apos;s average, in units over the chosen window. Values come from TreeSHAP on the LightGBM models.</p>
      </header>
      <SeriesPicker onChange={setPick} />
      <div className="row" role="group" aria-label="Window">
        {WINDOWS.map(([label], i) => <button key={label} className={i === w ? "" : "ghost"} onClick={() => setW(i)}>{label}</button>)}
      </div>
      {error && <p className="error">{error}</p>}
      {data && values && (
        <section className="panel">
          <h2>{data.series.item_id} at {data.series.store_id}: forecast {fmt(total, 1)} units</h2>
          <DriverBars values={values} />
          <p className="muted small">{data.drivers.method} SHAP values are additive in log space; the split into units is proportional, so totals are exact and individual bars approximate.</p>
        </section>
      )}
    </>
  );
}
