"""DAG integrity: both DAGs import and wire tasks in the documented order.

Runs where Airflow is installed (the Airflow image, or `uvx --from apache-airflow`).
"""

from pathlib import Path

import pytest

airflow = pytest.importorskip("airflow")
pytestmark = pytest.mark.integration

DAGS = Path(__file__).resolve().parents[2] / "orchestration" / "airflow" / "dags"


def test_dags_import_without_errors() -> None:
    from airflow.models import DagBag

    bag = DagBag(dag_folder=str(DAGS), include_examples=False)
    assert bag.import_errors == {}
    assert {"forecast_training", "nlp_training"} <= set(bag.dag_ids)


def test_forecast_dag_order() -> None:
    from airflow.models import DagBag

    dag = DagBag(dag_folder=str(DAGS), include_examples=False).get_dag("forecast_training")
    order = [t.task_id for t in dag.topological_sort()]
    assert order == [
        "validate",
        "profile",
        "backtest_baselines",
        "train_lightgbm",
        "evaluate",
        "compare_and_register",
        "publish_forecasts",
        "monitor",
    ]
