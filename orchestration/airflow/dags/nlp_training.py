"""Monthly NLP event-model pipeline (plan Phase 28).

ingest → validate → preprocess → baselines → fine-tune → evaluate → benchmark → register

Human-validated test labels are read from the review tool's store before evaluation;
without them the evaluation stops (teacher or rule labels are never test truth).
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

CLI = "cd /opt/demand-intel && demand-intel"

with DAG(
    dag_id="nlp_training",
    description="Recall corpus → event classifier → evaluation → benchmark → registry",
    schedule="@monthly",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "demand-intel", "retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["nlp"],
) as dag:
    ingest = BashOperator(task_id="ingest_and_validate", bash_command=f"{CLI} ingest text")
    preprocess = BashOperator(task_id="preprocess", bash_command=f"{CLI} nlp-prepare")
    baselines = BashOperator(
        task_id="baselines", bash_command=f"{CLI} nlp-train --models tfidf,embeddings"
    )
    finetune = BashOperator(
        task_id="fine_tune",
        bash_command=f"{CLI} nlp-train --models encoder",
        execution_timeout=timedelta(hours=4),
    )
    evaluate = BashOperator(task_id="evaluate", bash_command=f"{CLI} nlp-evaluate")
    benchmark = BashOperator(task_id="benchmark", bash_command=f"{CLI} nlp-benchmark")
    register = BashOperator(task_id="register", bash_command=f"{CLI} nlp-promote")

    ingest >> preprocess >> baselines >> finetune >> evaluate >> benchmark >> register
