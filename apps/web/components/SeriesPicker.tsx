"use client";
import { useEffect, useState } from "react";
import { useApi } from "./useApi";
import type { SeriesRow } from "@/lib/api";

const STORES = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"];
const CATS = ["FOODS", "HOBBIES", "HOUSEHOLD"];
const KEY = "di:series";

export type Pick = { item_id: string; store_id: string };

function remembered(): Pick {
  try {
    const v = localStorage.getItem(KEY);
    if (v) return JSON.parse(v) as Pick;
  } catch { /* storage unavailable */ }
  return { item_id: "FOODS_3_090", store_id: "CA_1" };
}

export default function SeriesPicker({ onChange }: { onChange: (p: Pick) => void }) {
  const [store, setStore] = useState("CA_1");
  const [cat, setCat] = useState("FOODS");
  const [item, setItem] = useState("FOODS_3_090");
  const [q, setQ] = useState("");
  useEffect(() => {
    const r = remembered();
    setStore(r.store_id); setItem(r.item_id); setCat(r.item_id.split("_")[0]);
    onChange(r);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const { data } = useApi<{ total: number; items: SeriesRow[] }>(
    `/v1/series?store_id=${store}&cat_id=${cat}&limit=1000${q ? `&q=${encodeURIComponent(q)}` : ""}`,
  );
  const choose = (p: Pick) => {
    setItem(p.item_id); setStore(p.store_id);
    try { localStorage.setItem(KEY, JSON.stringify(p)); } catch { /* ignore */ }
    onChange(p);
  };
  return (
    <div className="row">
      <label className="f" htmlFor="pick-store">Store
        <select id="pick-store" value={store} onChange={(e) => { setStore(e.target.value); choose({ item_id: item, store_id: e.target.value }); }}>
          {STORES.map((s) => <option key={s}>{s}</option>)}
        </select>
      </label>
      <label className="f" htmlFor="pick-cat">Category
        <select id="pick-cat" value={cat} onChange={(e) => setCat(e.target.value)}>
          {CATS.map((c) => <option key={c}>{c}</option>)}
        </select>
      </label>
      <label className="f" htmlFor="pick-q">Filter items
        <input id="pick-q" value={q} placeholder="e.g. 3_09" onChange={(e) => setQ(e.target.value)} />
      </label>
      <label className="f" htmlFor="pick-item">SKU ({data?.total ?? "…"})
        <select id="pick-item" value={item} onChange={(e) => choose({ item_id: e.target.value, store_id: store })}>
          {!data?.items.some((s) => s.item_id === item) && <option value={item}>{item}</option>}
          {data?.items.map((s) => <option key={s.id} value={s.item_id}>{s.item_id} · {s.demand_class}</option>)}
        </select>
      </label>
    </div>
  );
}
