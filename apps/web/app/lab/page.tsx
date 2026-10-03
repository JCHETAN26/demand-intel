"use client";
import { CompareBars } from "@/components/charts";
import { useApi } from "@/components/useApi";
import { fmt, pct } from "@/lib/api";

type Row = Record<string, string | number | boolean | null>;
type Lab = { model_comparison?: Row[]; forecast_ablation?: Row[]; paired_tests?: Row[]; nlp_results?: Row[];
  probabilistic_calibration?: Row[]; inventory_policy_differences?: Row[]; multimodal_ablation?: Row[];
  registry?: { tasks: Record<string, { champion: string; challengers: string[]; since: string | null }> } };

const n = (v: unknown) => (typeof v === "number" ? v : Number.NaN);

function Table({ rows, cols }: { rows: Row[]; cols: [string, string, (v: unknown) => string][] }) {
  return (
    <div className="tablewrap"><table>
      <thead><tr>{cols.map(([k, label]) => <th key={k}>{label}</th>)}</tr></thead>
      <tbody>{rows.map((r, i) => <tr key={i}>{cols.map(([k, , f]) => <td key={k} className={typeof r[k] === "number" ? "num" : ""}>{f(r[k])}</td>)}</tr>)}</tbody>
    </table></div>
  );
}

const txt = (v: unknown) => (v === null || v === undefined ? "—" : String(v));
const f3 = (v: unknown) => fmt(n(v), 3);
const p1 = (v: unknown) => pct(n(v));

export default function LabPage() {
  const { data, error } = useApi<Lab>("/v1/lab");
  const mc = data?.model_comparison ?? [];
  const pop = (p: string) => mc.filter((r) => r.population === p);
  const snaive = (p: string) => n(pop(p).find((r) => r.model === "seasonal_naive")?.backtest_wrmsse);
  return (
    <>
      <header>
        <h1>Model lab</h1>
        <p className="lede">Every model trained for the comparison, including the ones that lost. Lower WRMSSE is better; the dashed line marks seasonal naive, the baseline every model has to beat.</p>
      </header>
      {error && <p className="error">{error}</p>}
      {data?.registry && (
        <section className="tiles">
          {Object.entries(data.registry.tasks).map(([task, t]) => (
            <div className="tile" key={task}><span className="k">{task} champion</span><span className="v mono champion">{t.champion}</span>
              <span className="s">{t.challengers.length ? `challengers: ${t.challengers.join(", ")}` : "no open challengers"}</span></div>
          ))}
        </section>
      )}
      {(["full", "panel"] as const).map((p) => pop(p).length > 0 && (
        <section className="panel" key={p}>
          <h2>{p === "full" ? "All 30,490 series" : "3,000-series panel"}: backtest WRMSSE (5 folds)</h2>
          <CompareBars rows={pop(p).map((r) => ({ label: String(r.model), value: n(r.backtest_wrmsse) }))} metric="WRMSSE" reference={snaive(p)} />
          <Table rows={pop(p)} cols={[
            ["model", "Model", txt], ["backtest_wrmsse", "WRMSSE (folds)", f3], ["test_wrmsse", "WRMSSE (test)", f3],
            ["test_wape", "WAPE", p1], ["test_bias", "Bias", p1], ["test_wspl", "Pinball (scaled)", f3],
            ["seconds_per_fold", "Seconds/fold", (v) => fmt(n(v), 0)],
          ]} />
        </section>
      ))}
      {!!data?.paired_tests?.length && (
        <section className="panel"><h2>Paired comparisons (level-12 WRMSSE, folds 1–5)</h2>
          <Table rows={data.paired_tests} cols={[
            ["model_a", "Model A", txt], ["model_b", "vs B", txt], ["population", "Series", txt], ["diff", "Δ (A−B)", f3],
            ["ci_low", "CI low", f3], ["ci_high", "CI high", f3], ["fold_wins", "Fold wins", txt], ["improvement", "A improves", txt],
          ]} />
          <p className="muted small">An improvement needs the 95% bootstrap interval (series resampled) to lie entirely below zero.</p>
        </section>
      )}
      {!!data?.forecast_ablation?.length && (
        <section className="panel"><h2>Feature ablation (LightGBM, panel)</h2>
          <Table rows={data.forecast_ablation} cols={[["step", "Step", txt], ["model", "Model", txt], ["backtest_wrmsse", "WRMSSE", f3],
            ["delta_backtest_wrmsse", "Δ vs previous", f3], ["backtest_wape", "WAPE", p1], ["backtest_bias", "Bias", p1]]} />
        </section>
      )}
      {!!data?.nlp_results?.length && (
        <section className="panel"><h2>Event model (NLP), test split</h2>
          <Table rows={data.nlp_results.filter((r) => r.split === "test")} cols={[["model", "Model", txt], ["task", "Task", txt],
            ["truth", "Truth", txt], ["macro_f1", "Macro-F1", f3], ["accuracy", "Accuracy", p1], ["ece", "ECE", f3]]} />
          <p className="muted small">Truth &quot;weak&quot; is agreement with the labelling rules and is never used to choose a model; &quot;gold&quot; is the human-validated set.</p>
        </section>
      )}
      {!!data?.inventory_policy_differences?.length && (
        <section className="panel"><h2>Inventory policies: total cost differences</h2>
          <Table rows={data.inventory_policy_differences} cols={[["scenario", "Scenario", (v) => txt(v).replace(" (SCENARIO — not observed data)", "")], ["comparison", "Comparison", txt],
            ["relative", "Relative", p1], ["ci_low", "CI low", (v) => fmt(n(v), 0)], ["ci_high", "CI high", (v) => fmt(n(v), 0)]]} />
          <p className="muted small">L = lead time in days, p = stockout cost as a multiple of unit price, SL = target service level. Costs, lead times and stock are scenarios, not observed data: M5 has no inventory.</p>
        </section>
      )}
    </>
  );
}
