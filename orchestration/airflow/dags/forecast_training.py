"""Weekly forecasting pipeline (plan Phase 28).

validate → profile → backtest → train → evaluate → compare champion → register → publish

Every task calls the `demand-intel` CLI, so the DAG orchestrates the same code that runs
locally and in CI. Promotion is gated (spec §11.2): `promote` records a decision either
way, and `publish` rebuilds the serving bundle from the registered champion.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

CLI = "cd /opt/demand-intel && demand-intel"

default_args = {
    "owner": "demand-intel",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="forecast_training",
    description="Validate data, backtest, train, gate and publish the demand forecaster",
    schedule="@weekly",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["forecasting"],
) as dag:
    validate = BashOperator(
        task_id="validate",
        bash_command=f"{CLI} ingest m5 && {CLI} ingest weather",
    )
    profile = BashOperator(task_id="profile", bash_command=f"{CLI} eda")
    backtest_baselines = BashOperator(
        task_id="backtest_baselines",
        bash_command="cd /opt/demand-intel && scripts/run_forecasting.sh baselines statistical",
        execution_timeout=timedelta(hours=3),
    )
    train = BashOperator(
        task_id="train_lightgbm",
        bash_command="cd /opt/demand-intel && FORCE=1 scripts/run_forecasting.sh lgbm_full",
        execution_timeout=timedelta(hours=3),
    )
    evaluate = BashOperator(task_id="evaluate", bash_command=f"{CLI} report")
    gate = BashOperator(
        task_id="compare_and_register",
        bash_command=f"{CLI} promote --challenger lgbm --population full",
    )
    publish = BashOperator(
        task_id="publish_forecasts",
        bash_command=f"{CLI} forecast-batch",
        execution_timeout=timedelta(hours=1),
    )
    monitor = BashOperator(task_id="monitor", bash_command=f"{CLI} monitor")

    validate >> profile >> backtest_baselines >> train >> evaluate >> gate >> publish >> monitor
