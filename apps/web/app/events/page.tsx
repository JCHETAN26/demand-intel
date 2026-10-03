"use client";
import { useState } from "react";
import { useApi } from "@/components/useApi";
import { post, pct, type EventRow } from "@/lib/api";

type Extracted = { event_type: string; confidence: number; severity: string | null; direction: string;
  category: string; states: string[]; nationwide: boolean; model: string };

const SEV: Record<string, string> = { high: "bad", medium: "warn", low: "good" };

export default function Events() {
  const [state, setState] = useState("");
  const [type, setType] = useState("");
  const { data, error } = useApi<{ total: number; counts: Record<string, number>; items: EventRow[] }>(
    `/v1/events/recent?limit=60${state ? `&state=${state}` : ""}${type ? `&event_type=${type}` : ""}`,
  );
  const [text, setText] = useState("Product may be contaminated with Listeria monocytogenes. Distributed in CA, TX and WI.");
  const [res, setRes] = useState<Extracted | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const run = async () => {
    setErr(null);
    try { setRes(await post<Extracted>("/v1/events/extract", { text })); } catch (e) { setErr((e as Error).message); }
  };
  return (
    <>
      <header>
        <h1>External intelligence</h1>
        <p className="lede">FDA food recall reports from the 12 months before the forecast origin, with the hazard, severity and states the event model extracted. Each card shows the source text so you can check the extraction.</p>
      </header>
      <section className="panel">
        <h2>Try the extractor</h2>
        <label className="f" htmlFor="ev-text">Recall text<textarea id="ev-text" value={text} onChange={(e) => setText(e.target.value)} /></label>
        <div className="row"><button onClick={run}>Extract event</button>{err && <span className="error">{err}</span>}</div>
        {res && (
          <div className="tablewrap"><table><tbody>
            <tr><th>Type</th><td>{res.event_type}</td></tr>
            <tr><th>Direction</th><td>{res.direction}</td></tr>
            <tr><th>Severity</th><td>{res.severity ?? "not predicted by this model"}</td></tr>
            <tr><th>Confidence</th><td>{pct(res.confidence, 0)}</td></tr>
            <tr><th>States</th><td>{res.nationwide ? "Nationwide" : res.states.join(", ") || "none named"}</td></tr>
            <tr><th>Model</th><td className="mono">{res.model}</td></tr>
          </tbody></table></div>
        )}
      </section>
      <section className="panel">
        <div className="row">
          <label className="f" htmlFor="ev-state">State<select id="ev-state" value={state} onChange={(e) => setState(e.target.value)}>
            <option value="">All</option><option>CA</option><option>TX</option><option>WI</option></select></label>
          <label className="f" htmlFor="ev-type">Event type<select id="ev-type" value={type} onChange={(e) => setType(e.target.value)}>
            <option value="">All</option>{Object.keys(data?.counts ?? {}).map((t) => <option key={t}>{t}</option>)}</select></label>
          <span className="muted small">{data ? `${data.total.toLocaleString()} reports` : ""}</span>
        </div>
        {error && <p className="error">{error}</p>}
        {data?.items.map((e) => (
          <article key={e.doc_id} className="event">
            <div className="row">
              <b>{e.event_type.replaceAll("_", " ").toLowerCase()}</b>
              <span className={`pill ${SEV[e.fda_severity] ?? ""}`}><span className="dot" />FDA {e.fda_severity}</span>
              <span className="muted small">{e.report_date} · {e.nationwide ? "nationwide" : e.states.join(", ")} · confidence {pct(e.confidence, 0)} · {e.model}</span>
            </div>
            <div>{e.reason}</div>
            <div className="muted small">{e.product}</div>
          </article>
        ))}
      </section>
    </>
  );
}
