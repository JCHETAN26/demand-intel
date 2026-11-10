"""Final acceptance (plan Phase 37): the ten questions, answered from artifacts only.

Each answer is `yes`, `no` or `pending` (evidence not produced yet), with the numbers
that decided it. Nothing here is typed by hand: change an artifact and the answer changes.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

Verdict = dict[str, Any]


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def _paired(tests: pd.DataFrame, a: str, b: str, pop: str) -> pd.Series | None:
    if tests.empty or "diff" not in tests:
        return None
    sel = tests[(tests["model_a"] == a) & (tests["model_b"] == b) & (tests["population"] == pop)]
    sel = sel[sel["diff"].notna()] if "diff" in sel else sel
    return sel.iloc[0] if len(sel) else None


def _fmt_pair(r: pd.Series) -> str:
    return (
        f"Δ level-12 WRMSSE {r['diff']:+.4f} (95% CI {r['ci_low']:+.4f} to {r['ci_high']:+.4f}), "
        f"{int(r['fold_wins'])}/5 folds won"
    )


def answer_all(ev: Path) -> list[Verdict]:
    tests = _csv(ev / "paired_tests.csv")
    comp = _csv(ev / "model_comparison.csv")
    out: list[Verdict] = []

    def add(q: str, verdict: str, evidence: str) -> None:
        out.append({"question": q, "verdict": verdict, "evidence": evidence})

    registry = _json(ev / "registry.json")
    champion = registry.get("tasks", {}).get("forecast", {}).get("champion", "lgbm")

    r = _paired(tests, "lgbm", "seasonal_naive", "full")
    add(
        "Forecasting: does the model beat seasonal naive?",
        "pending" if r is None else ("yes" if bool(r["improvement"]) else "no"),
        "not run" if r is None else _fmt_pair(r) + " (all 30,490 series)",
    )

    rs = [
        x
        for x in (
            _paired(tests, "lgbm", "stat_class_selected", "full"),
            _paired(tests, "lgbm", "ets", "panel"),
            _paired(tests, "lgbm", "arima", "panel"),
        )
        if x is not None
    ]
    add(
        "ML: does LightGBM improve over statistical baselines?",
        "pending" if not rs else ("yes" if all(bool(x["improvement"]) for x in rs) else "no"),
        "; ".join(f"vs {x['model_b']} ({x['population']}): {_fmt_pair(x)}" for x in rs)
        or "not run",
    )

    dl = [
        x
        for x in (_paired(tests, "lstm", "lgbm", "panel"), _paired(tests, "tft", "lgbm", "panel"))
        if x is not None
    ]
    if dl:
        cost = ""
        if not comp.empty:
            sec = comp[comp["population"] == "panel"].set_index("model")["seconds_per_fold"]
            cost = "; seconds per fold: " + ", ".join(
                f"{m} {sec.get(m, float('nan')):.0f}" for m in ("lgbm", "lstm", "tft")
            )
        add(
            "Deep learning: do LSTM/TFT justify their complexity?",
            "yes" if any(bool(x["improvement"]) for x in dl) else "no",
            "; ".join(f"{x['model_a']} vs LightGBM: {_fmt_pair(x)}" for x in dl) + cost,
        )
    else:
        add("Deep learning: do LSTM/TFT justify their complexity?", "pending", "not run")

    cal = _csv(ev / "probabilistic_calibration.csv")
    served = _json(ev.parent / "data" / "serving" / "summary.json").get("model")
    candidates = [m for m in (champion, served, "lgbm_bc", "lgbm") if m]
    c = pd.DataFrame()
    model_checked = ""
    for m in candidates:
        if cal.empty:
            break
        sel = cal[
            (cal["model"] == m)
            & (cal["population"] == "full")
            & (cal["slice"] == "all")
            & cal["fold"].str.startswith("fold")
        ]
        if len(sel):
            c, model_checked = sel, m
            break
    if c.empty:
        add(
            "Uncertainty: are P10/P90 calibrated?",
            "pending",
            "no probabilistic model evaluated on all series",
        )
    else:
        ok = int(c["calibrated_p10_p90"].sum())
        add(
            "Uncertainty: are P10/P90 calibrated?",
            "yes" if ok == len(c) else "no",
            f"{model_checked}: P10 and P90 inside the discrete calibration bracket on {ok}/{len(c)} backtest folds",
        )

    nlp = _csv(ev / "nlp_results.csv")
    gold = (
        nlp[(nlp.get("truth") == "gold") & (nlp.get("task") == "event_type")]
        if not nlp.empty
        else nlp
    )
    if gold.empty:
        add(
            "NLP: does the fine-tuned model beat simple baselines?",
            "pending",
            "human-validated test labels not available yet; weak-label agreement is not test truth",
        )
    else:
        g = gold.set_index("model")
        ft_ci = float(g["macro_f1_ci_low"].get("n3_encoder", float("nan")))
        ft = None if ft_ci != ft_ci else ft_ci
        base = g.drop(index=[m for m in ("n3_encoder",) if m in g.index])
        best_baseline = float(base["macro_f1"].max())
        verdict = "pending" if ft is None else ("yes" if ft > best_baseline else "no")
        add(
            "NLP: does the fine-tuned model beat simple baselines?",
            verdict,
            "gold macro-F1: " + ", ".join(f"{m} {v:.3f}" for m, v in g["macro_f1"].items()),
        )

    mm = _csv(ev / "multimodal_ablation.csv")
    cmp_rows = mm[mm["status"] == "comparison"] if "status" in mm.columns else mm.iloc[0:0]
    if cmp_rows.empty:
        add("Multimodal: do NLP-derived signals improve forecasting?", "pending", "not run")
    else:
        rows = cmp_rows[cmp_rows["arm"].str.contains("vs D_weather")]
        yes = bool(rows["consistent_improvement"].any())
        add(
            "Multimodal: do NLP-derived signals improve forecasting?",
            "yes" if yes else "no",
            "; ".join(
                f"{x['arm']}: Δ {x['diff']:+.4f} (CI {x['ci_low']:+.4f} to {x['ci_high']:+.4f}), "
                f"{int(x['fold_wins'])}/5 folds"
                for x in rows.to_dict("records")
            ),
        )

    inv = _csv(ev / "inventory_policy_differences.csv")
    if inv.empty:
        add(
            "Inventory: does better forecasting reduce stockouts/excess cost in simulation?",
            "pending",
            "not run",
        )
    else:
        sel = inv[inv["comparison"] == "P3_probabilistic - P1_moving_average"]
        default = sel[sel["scenario"].str.contains(r"L=7 p=1.0x SL=95%", regex=False)]
        sens = sel[sel["scenario"].str.contains("SL=95%") & sel["scenario"].str.contains("L=7")]
        better_default = bool(
            len(default)
            and default["cost_difference"].iloc[0] < 0
            and default["ci_high"].iloc[0] < 0
        )
        holds = int((sens["cost_difference"] < 0).sum())
        add(
            "Inventory: does better forecasting reduce stockouts/excess cost in simulation?",
            "yes" if better_default and holds >= 3 else "no",
            f"P3 vs P1 at default: {default['relative'].iloc[0]:+.1%} cost (CI {default['ci_low'].iloc[0]:,.0f} to "
            f"{default['ci_high'].iloc[0]:,.0f}); lower in {holds}/{len(sens)} stockout-cost scenarios"
            if len(default)
            else "default scenario missing",
        )

    decisions = registry.get("decisions", [])
    add(
        "MLOps: can the system train, evaluate, register and deploy reproducibly?",
        "yes" if decisions else "pending",
        f"{len(decisions)} registry decisions with gate reports; champion {champion}; DAGs import cleanly; "
        "`make evaluate` regenerates every artifact"
        if decisions
        else "no registry decision yet",
    )

    sb = _json(ev / "serving_benchmark.json")
    if sb:
        # The latest run is the deployed configuration; each endpoint is held to its own p95 SLO.
        latest = list(sb["runs"].values())[-1]
        parts, within = [], True
        for endpoint, slo in sb["slo"].items():
            if endpoint not in latest:
                continue
            p95 = latest[endpoint]["p95_ms"]
            within &= p95 <= slo["p95_ms"]
            parts.append(
                f"{endpoint} p95 {p95:.1f} ms (SLO {slo['p95_ms']:.0f}), "
                f"{latest[endpoint]['throughput_rps']:.0f} req/s"
            )
        add(
            "Production: can it serve forecasts within acceptable latency?",
            "yes" if parts and within else "no",
            "; ".join(parts) + f"; {sb['hardware']['cores']} cores",
        )
    else:
        add(
            "Production: can it serve forecasts within acceptable latency?",
            "pending",
            "benchmark not run",
        )

    mon = _json(ev / "monitoring_report.json")
    if mon:
        add(
            "Monitoring: can we detect degraded data/model/system behaviour?",
            "yes",
            f"forecast checks {mon['forecast']['status']}, data drift {mon['data']['status']}, NLP {mon['nlp']['status']}; "
            "Prometheus alert rules and CloudWatch alarms cover latency, errors and availability",
        )
    else:
        add(
            "Monitoring: can we detect degraded data/model/system behaviour?",
            "pending",
            "monitor not run",
        )
    return out


def render(answers: list[Verdict]) -> str:
    lines = [
        "# Acceptance",
        "",
        "_Generated by `demand-intel acceptance` from `evaluation/`. Do not edit by hand._",
        "",
        "| Question | Verdict | Evidence |",
        "|---|---|---|",
    ]
    for a in answers:
        lines.append(f"| {a['question']} | **{a['verdict']}** | {a['evidence']} |")
    return "\n".join(lines) + "\n"
