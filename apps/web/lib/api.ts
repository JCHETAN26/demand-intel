// Thin client for the Demand Intelligence API. All numbers shown in the app come from it.
export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export type Summary = {
  origin_date: string;
  forecast_dates: [string, string];
  horizon_days: number;
  skus_monitored: number;
  stockout_risk: number;
  excess_inventory_risk: number;
  risk_definition: string;
  evaluation: Record<string, number>;
  model: string;
};

export type SeriesRow = {
  id: string; item_id: string; dept_id: string; cat_id: string;
  store_id: string; state_id: string; demand_class: string;
};

export type SeriesDetail = {
  series: SeriesRow & { volume_tercile: string };
  history: { dates: string[]; sales: number[]; price: (number | null)[] };
  forecast: { dates: string[]; mean: number[]; p05: number[]; p10: number[]; p25: number[]; p50: number[];
    p75: number[]; p90: number[]; p95: number[]; price: (number | null)[] };
  drivers: { families: string[]; daily_units: number[][]; week1_units: Record<string, number>; method: string };
};

export type RiskRow = { item_id: string; store_id: string; cat_id: string; scenario_stock: number;
  lead_time_p95: number; gap: number };

export type EventRow = { doc_id: string; report_date: string; reason: string; product: string;
  distribution: string; states: string[]; nationwide: boolean; fda_severity: string;
  event_type: string; confidence: number; model: string };

export type Recommendation = { lead_time_days: number; service_level: number; expected_demand: number;
  p50: number; p90: number; target_level: number; safety_stock: number; inventory_position: number;
  recommended_order: number; stockout_probability: number; rho: number; note: string };

export const fmt = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
export const pct = (v: number | null | undefined, d = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`;
