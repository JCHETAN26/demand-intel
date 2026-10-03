# Orchestration

Two DAGs run the project's CLI end to end:

| DAG | Schedule | Steps |
|---|---|---|
| `forecast_training` | weekly | validate → profile → backtest → train → evaluate → gate/register → publish → monitor |
| `nlp_training` | monthly | ingest/validate → preprocess → baselines → fine-tune → evaluate → benchmark → register |

Run locally with the compose profile:

```bash
docker compose --profile airflow up -d airflow
# UI: http://localhost:8080 (standalone mode prints the admin password in the logs)
```

`tests/integration/test_dags.py` checks that both DAGs import and have the expected task order.
