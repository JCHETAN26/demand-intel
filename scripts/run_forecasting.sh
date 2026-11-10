#!/usr/bin/env bash
# Every forecasting backtest behind evaluation/backtest/ (PROJECT_SPEC.md §5.3, §6).
# Long-running on a 4-core machine; each step is skipped when its results file exists
# unless FORCE=1. Usage: scripts/run_forecasting.sh [step ...]
set -euo pipefail
cd "$(dirname "$0")/.."
run() {  # run <population> <folds> <model...>
  local pop=$1 folds=$2; shift 2
  for m in "$@"; do
    local out="evaluation/backtest/${m}__${pop}.csv"
    if [[ "${FORCE:-0}" != 1 && -f "$out" ]] && grep -q ",test," "$out" && grep -q ",fold_5," "$out"; then
      echo "skip $m $pop (results exist)"; continue
    fi
    echo "== $m $pop $folds $(date -u +%H:%M:%S)"
    uv run demand-intel backtest "$m" --population "$pop" --folds "$folds"
  done
}
steps=("${@:-baselines statistical lgbm_full lgbm_bc lgbm_panel ablation deep weather_oracle multimodal}")
for step in ${steps[@]}; do
  case $step in
    baselines) run full all naive seasonal_naive moving_average_28
               run panel all naive seasonal_naive moving_average_28 ;;
    statistical) run full all ses_seasonal croston tsb stat_class_selected
                 run panel all ses_seasonal croston tsb stat_class_selected ets arima ;;
    lgbm_full) run full all lgbm ;;
    lgbm_panel) run panel all lgbm ;;
    lgbm_bc) run full all lgbm_bc ;;  # spec v0.5 bias-corrected challenger
    ablation) run panel all lgbm_abl1_sales lgbm_abl2_calendar lgbm_abl3_snap lgbm_abl4_price \
                            lgbm_abl5_hierarchy lgbm_abl6_weather ;;
    deep) run panel all lstm tft ;;
    weather_oracle) run panel all lgbm_weather_oracle ;;
    multimodal) run panel all lgbm_mm_E_text_embed lgbm_mm_F_text_events ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
