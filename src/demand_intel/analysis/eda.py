"""Exploratory demand analysis (plan Phase 7) and extra data-quality checks (Phase 6).

`run_eda` writes CSV/JSON artifacts to `evaluation/eda/` and a report whose numbers are all
read from those artifacts. Effects here are *descriptive*: they motivate hypotheses that the
forecasting experiments test, they are not causal estimates.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from demand_intel.analysis.profile import PROFILE_END_DAY, series_profile, suspicious_zero_runs
from demand_intel.data import M5Data, first_sale_index

CATEGORIES = ("FOODS", "HOBBIES", "HOUSEHOLD")


@dataclass
class EdaResult:
    tables: dict[str, pd.DataFrame]
    overview: dict[str, Any]
    hypotheses: list[dict[str, Any]]


def _category_daily(m5: M5Data) -> pd.DataFrame:
    """Total units per category × day, joined to the calendar."""
    cat = m5.series["cat_id"].to_numpy()
    totals = {c: m5.sales[cat == c].sum(axis=0) for c in CATEGORIES}
    df = pd.DataFrame(totals)
    df["d"] = np.arange(1, m5.n_days + 1)
    return df.merge(m5.calendar, on="d")


def _weekly_matrices(m5: M5Data) -> tuple[np.ndarray, np.ndarray]:
    """Weekly units and weekly price per series (weeks fully inside the sales period)."""
    cal = m5.calendar[m5.calendar["d"] <= m5.n_days]
    weeks, week_idx = np.unique(cal["wm_yr_wk"].to_numpy(), return_inverse=True)
    full = np.bincount(week_idx) == 7
    onehot = np.zeros((m5.n_days, len(weeks)), dtype=np.float32)
    onehot[np.arange(m5.n_days), week_idx] = 1.0
    units = (m5.sales.astype(np.float32) @ onehot)[:, full]
    assert m5.prices is not None
    first_day_of_week = np.array([np.flatnonzero(week_idx == w)[0] for w in range(len(weeks))])
    price = m5.prices[:, first_day_of_week][:, full]
    return units, price


def price_elasticity(m5: M5Data) -> pd.DataFrame:
    """Within-series log-log slope of weekly units on price, by category.

    Two-way demeaning (series, then week within category) removes level differences and
    common seasonality. Only series-weeks with a price and positive sales enter, so the slope
    describes the intensive margin. Descriptive only: prices are not randomized.
    """
    units, price = _weekly_matrices(m5)
    rows = []
    for c in CATEGORIES:
        mask = (m5.series["cat_id"] == c).to_numpy()
        u, p = units[mask], price[mask]
        ok = (u > 0) & ~np.isnan(p) & (p > 0)
        y = np.where(ok, np.log(np.where(ok, u, 1)), np.nan)
        x = np.where(ok, np.log(np.where(ok, p, 1)), np.nan)
        varies = np.nanstd(x, axis=1) > 1e-6
        y, x = y[varies], x[varies]
        for axis in (1, 0, 1):
            y = y - np.nanmean(y, axis=axis, keepdims=True)
            x = x - np.nanmean(x, axis=axis, keepdims=True)
        sel = ~np.isnan(x) & ~np.isnan(y)
        slope = float((x[sel] * y[sel]).sum() / (x[sel] ** 2).sum())
        resid = y[sel] - slope * x[sel]
        # cluster-robust (by series) standard error
        g = np.repeat(np.arange(len(x)), x.shape[1]).reshape(x.shape)[sel]
        scores = np.bincount(g, weights=x[sel] * resid)
        se = float(np.sqrt((scores**2).sum()) / (x[sel] ** 2).sum())
        rows.append(
            {
                "cat_id": c,
                "elasticity": slope,
                "se": se,
                "series_with_price_variation": int(varies.sum()),
                "series_weeks": int(sel.sum()),
            }
        )
    return pd.DataFrame(rows)


def snap_effect(daily: pd.DataFrame, m5: M5Data) -> pd.DataFrame:
    """SNAP-day / non-SNAP-day mean units per state and category, plus FOODS vs HOUSEHOLD.

    SNAP days are concentrated early in the month, so the raw ratio mixes SNAP with
    day-of-month effects; the FOODS/HOUSEHOLD ratio-of-ratios nets out what both share.
    """
    rows = []
    for state in ("CA", "TX", "WI"):
        mask = (m5.series["state_id"] == state).to_numpy()
        cat = m5.series["cat_id"].to_numpy()
        snap = daily[f"snap_{state}"].to_numpy() == 1
        open_days = daily["d"].isin(_open_days(m5)).to_numpy()
        ratios = {}
        for c in CATEGORIES:
            tot = m5.sales[mask & (cat == c)].sum(axis=0)
            ratios[c] = tot[snap & open_days].mean() / tot[~snap & open_days].mean()
        rows.append(
            {
                "state_id": state,
                **{f"ratio_{c}": v for c, v in ratios.items()},
                "foods_vs_household": ratios["FOODS"] / ratios["HOUSEHOLD"],
            }
        )
    return pd.DataFrame(rows)


def _open_days(m5: M5Data) -> np.ndarray:
    """Days on which stores trade (total sales above 10% of the median day)."""
    total = m5.sales.sum(axis=0)
    return np.flatnonzero(total > 0.1 * np.median(total)) + 1


def event_effect(daily: pd.DataFrame) -> pd.DataFrame:
    """Units on each event day vs the mean of the same weekday ±1 and ±2 weeks (non-event)."""
    daily = daily.sort_values("d").reset_index(drop=True)
    units = daily[list(CATEGORIES)].to_numpy(dtype=float)
    is_event = daily["event_name_1"].notna().to_numpy()
    rows = []
    for i in np.flatnonzero(is_event):
        ref = [j for j in (i - 14, i - 7, i + 7, i + 14) if 0 <= j < len(daily) and not is_event[j]]
        if len(ref) < 2:
            continue
        ratio = units[i] / units[ref].mean(axis=0)
        rows.append(
            {
                "event_name": daily["event_name_1"].iloc[i],
                "event_type": daily["event_type_1"].iloc[i],
                **dict(zip(CATEGORIES, ratio, strict=True)),
            }
        )
    per_day = pd.DataFrame(rows)
    return (
        per_day.groupby(["event_type", "event_name"])[list(CATEGORIES)]
        .median()
        .round(3)
        .reset_index()
        .sort_values(["event_type", "event_name"])
    )


def run_eda(m5: M5Data) -> EdaResult:
    daily = _category_daily(m5)
    open_days = _open_days(m5)
    trading = daily[daily["d"].isin(open_days)]

    profile = series_profile(m5)
    zr = suspicious_zero_runs(m5)
    zr_poisson = suspicious_zero_runs(m5, method="poisson")
    first = first_sale_index(m5.sales)
    after_first = np.arange(m5.n_days)[None, :] >= first[:, None]
    assert m5.prices is not None
    unpriced_sales = (m5.sales > 0) & np.isnan(m5.prices[:, : m5.n_days])

    weekday = trading.groupby("weekday")[list(CATEGORIES)].mean()
    weekday = (weekday / weekday.mean()).reindex(
        ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    )
    month = trading.groupby("month")[list(CATEGORIES)].mean()
    month = month / month.mean()
    yearly = trading[trading["year"].between(2011, 2015)].groupby("year")[list(CATEGORIES)].mean()

    store_cat = (
        pd.DataFrame(
            {
                "store_id": m5.series["store_id"],
                "cat_id": m5.series["cat_id"],
                "units": m5.sales[:, -365:].sum(axis=1),
            }
        )
        .groupby(["store_id", "cat_id"])["units"]
        .sum()
        .unstack()
    )
    store_cat["total"] = store_cat.sum(axis=1)

    classes = profile.groupby(["cat_id", "demand_class"]).size().unstack(fill_value=0).astype(int)
    class_share = classes.div(classes.sum(axis=1), axis=0)

    elasticity = price_elasticity(m5)
    snap = snap_effect(daily, m5)
    events = event_effect(daily)

    closed = sorted(set(range(1, m5.n_days + 1)) - set(open_days.tolist()))
    overview: dict[str, Any] = {
        "series": len(m5.series),
        "days": m5.n_days,
        "profile_end_day": PROFILE_END_DAY,
        "zero_share_all_cells": float((m5.sales == 0).mean()),
        "zero_share_after_first_sale": float((m5.sales[after_first] == 0).mean()),
        "share_cells_before_first_sale": float((~after_first).mean()),
        "closed_days": closed,
        "closed_dates": m5.calendar.set_index("d").loc[closed, "date"].astype(str).tolist(),
        "sales_on_unpriced_days": int(unpriced_sales.sum()),
        "suspicious_zero_runs": int((zr.runs["kind"] == "stockout").sum()),
        "dormant_zero_runs": int((zr.runs["kind"] == "dormant").sum()),
        "dormant_day_share": zr.dormant_days / float(after_first.sum()),
        "suspicious_zero_run_days": zr.flagged_days,
        "suspicious_zero_run_series": zr.series_affected,
        "suspicious_zero_run_day_share": zr.flagged_days / float(after_first.sum()),
        "suspicious_zero_run_day_share_poisson": zr_poisson.flagged_days / float(after_first.sum()),
        "dormant_day_share_poisson": zr_poisson.dormant_days / float(after_first.sum()),
        "demand_classes": profile["demand_class"].value_counts().to_dict(),
    }
    tables = {
        "eda_weekday_index": weekday.round(4).reset_index(),
        "eda_month_index": month.round(4).reset_index(),
        "eda_yearly_mean_daily_units": yearly.round(1).reset_index(),
        "eda_store_category_units_last365": store_cat.reset_index(),
        "eda_demand_class_share": class_share.round(4).reset_index(),
        "eda_price_elasticity": elasticity.round(4),
        "eda_snap_effect": snap.round(4),
        "eda_event_effect": events,
    }
    hypotheses = _hypotheses(overview, tables)
    return EdaResult(
        tables={**tables, "_profile": profile, "_zero_runs": zr.runs},
        overview=overview,
        hypotheses=hypotheses,
    )


def _hypotheses(ov: dict[str, Any], t: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Turn the measured patterns into testable hypotheses, each tied to an experiment."""
    hs: list[dict[str, Any]] = []

    el = t["eda_price_elasticity"].set_index("cat_id")["elasticity"].astype(float)
    most, least = str(el.idxmin()), str(el.idxmax())
    hs.append(
        {
            "id": "H1",
            "statement": f"Price sensitivity differs by category: {most} demand responds most "
            f"to price (within-series elasticity {el[most]:.2f}) and {least} least "
            f"({el[least]:.2f}).",
            "test": "Feature ablation step 4 (+price) gives the largest WRMSSE gain in "
            f"{most}; SHAP price attributions are largest in {most}.",
        }
    )

    cs = t["eda_demand_class_share"].set_index("cat_id")
    cs = cs.reindex(columns=["intermittent", "lumpy"], fill_value=0.0)
    sparse = (cs["intermittent"] + cs["lumpy"]).sort_values()
    hs.append(
        {
            "id": "H2",
            "statement": f"Most series are intermittent: {sparse.iloc[-1]:.0%} of "
            f"{sparse.index[-1]} and {sparse.iloc[0]:.0%} of {sparse.index[0]} series are "
            "intermittent or lumpy (Syntetos–Boylan).",
            "test": "Croston/TSB beat ETS on intermittent/lumpy series; a Tweedie or "
            "Poisson LightGBM objective beats L2 on them.",
        }
    )

    wk = t["eda_weekday_index"].set_index("weekday")
    amp = (wk.max() / wk.min()).sort_values()
    hs.append(
        {
            "id": "H3",
            "statement": "All categories have a strong weekly cycle peaking at the weekend; "
            f"peak/trough ratio ranges from {amp.iloc[0]:.2f} ({amp.index[0]}) to "
            f"{amp.iloc[-1]:.2f} ({amp.index[-1]}).",
            "test": "Seasonal naive (7) beats naive (1) at every horizon; calendar features "
            "give a material gain in ablation step 2.",
        }
    )

    snap = t["eda_snap_effect"]
    hs.append(
        {
            "id": "H4",
            "statement": "SNAP days lift FOODS more than HOUSEHOLD in every state "
            f"(FOODS/HOUSEHOLD SNAP ratio-of-ratios {snap['foods_vs_household'].min():.3f}"
            f"–{snap['foods_vs_household'].max():.3f})."
            if (snap["foods_vs_household"] > 1).all()
            else "SNAP effects on FOODS relative to HOUSEHOLD are not consistent across states.",
            "test": "Ablation step 3 (+SNAP) improves FOODS WRMSSE but not HOUSEHOLD/HOBBIES.",
        }
    )

    hs.append(
        {
            "id": "H5",
            "statement": f"{ov['suspicious_zero_run_series']:,} series contain "
            f"{ov['suspicious_zero_runs']:,} implausible 14–55-day zero runs "
            f"({ov['suspicious_zero_run_day_share']:.2%} of active series-days), consistent "
            "with unobserved stockouts; a further "
            f"{ov['dormant_day_share']:.2%} of active series-days sit in dormant runs of "
            "56+ days, which are kept as real demand regimes.",
            "test": "Headline metrics are re-run with these days masked (spec §2.1); model "
            "rankings should not change.",
        }
    )

    yr = t["eda_yearly_mean_daily_units"].set_index("year")
    growth = (yr.iloc[-1] / yr.iloc[0]) ** (1 / (len(yr) - 1)) - 1
    hs.append(
        {
            "id": "H6",
            "statement": "Demand trends upward 2011–2015 in every category "
            f"(CAGR {growth.min():.1%}–{growth.max():.1%}), partly through assortment "
            "growth (new items entering)"
            if (growth > 0).all()
            else "Category trends differ in sign over 2011–2015.",
            "test": "Models with level adaptation (ETS, rolling features) beat long-window "
            "averages; bias stays near zero on the test window.",
        }
    )
    return hs


def write_eda(result: EdaResult, out_dir: Path, data_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in result.tables.items():
        if name.startswith("_"):
            continue
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        written.append(path)
    (out_dir / "eda_overview.json").write_text(json.dumps(result.overview, indent=2, default=int))
    (out_dir / "eda_hypotheses.json").write_text(json.dumps(result.hypotheses, indent=2))
    data_dir.mkdir(parents=True, exist_ok=True)
    result.tables["_profile"].to_parquet(data_dir / "series_profile.parquet", index=False)
    result.tables["_zero_runs"].to_parquet(data_dir / "suspicious_zero_runs.parquet", index=False)
    return written


def render_report(eda_dir: Path) -> str:
    """Markdown report built only from the artifacts in `eda_dir`."""
    ov = json.loads((eda_dir / "eda_overview.json").read_text())
    hs = json.loads((eda_dir / "eda_hypotheses.json").read_text())

    def table(name: str) -> str:
        df = pd.read_csv(eda_dir / f"{name}.csv")
        ints = df.select_dtypes("integer").columns
        df[ints] = df[ints].astype(str)  # keep years/months/counts out of float formatting
        return df.to_markdown(index=False, floatfmt=".3f")

    lines = [
        "# Exploratory Demand Analysis",
        "",
        "_Generated by `demand-intel eda` from `evaluation/eda/`. Do not edit by hand._",
        "",
        "## Overview",
        "",
        f"- Series: {ov['series']:,}; days: {ov['days']:,}.",
        f"- Zero share of all series-days: {ov['zero_share_all_cells']:.1%}; after each "
        f"series' first sale: {ov['zero_share_after_first_sale']:.1%}; "
        f"{ov['share_cells_before_first_sale']:.1%} of cells precede a series' first sale.",
        f"- Store-closure days (total sales < 10% of median): {', '.join(ov['closed_dates'])}.",
        f"- Units sold on days without a price: {ov['sales_on_unpriced_days']:,}.",
        f"- Suspicious zero runs of 14–55 days (likely stockouts, spec §2.1): "
        f"{ov['suspicious_zero_runs']:,} runs in {ov['suspicious_zero_run_series']:,} series, "
        f"{ov['suspicious_zero_run_day_share']:.2%} of active series-days "
        f"(Poisson test: {ov['suspicious_zero_run_day_share_poisson']:.2%}).",
        f"- Dormant zero runs of 56+ days: {ov['dormant_zero_runs']:,} runs, "
        f"{ov['dormant_day_share']:.2%} of active series-days (not masked).",
        f"- Demand classes (profiling window to d_{ov['profile_end_day']}): "
        + ", ".join(f"{k} {v:,}" for k, v in ov["demand_classes"].items())
        + ".",
        "",
        "## Hypotheses",
        "",
        *[f"**{h['id']}.** {h['statement']}  \n_Test:_ {h['test']}\n" for h in hs],
        "## Demand classes by category (share of series)",
        "",
        table("eda_demand_class_share"),
        "",
        "## Weekly cycle (mean units by weekday / overall mean)",
        "",
        table("eda_weekday_index"),
        "",
        "## Annual cycle (mean units by month / overall mean)",
        "",
        table("eda_month_index"),
        "",
        "## Trend (mean daily units per year)",
        "",
        table("eda_yearly_mean_daily_units"),
        "",
        "## Price response (within-series log-log slope, descriptive)",
        "",
        table("eda_price_elasticity"),
        "",
        "## SNAP days (mean units SNAP / non-SNAP)",
        "",
        table("eda_snap_effect"),
        "",
        "## Stores × category (units, last 365 days)",
        "",
        table("eda_store_category_units_last365"),
        "",
        "## Events (median units vs same weekday ±1–2 weeks)",
        "",
        table("eda_event_effect"),
        "",
    ]
    return "\n".join(lines)
