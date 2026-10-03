# Airflow pins most of its dependency tree, so it gets its own image with the project
# installed alongside (spec: orchestration runs the same CLI as everything else).
FROM apache/airflow:2.10.5-python3.11
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
USER airflow
COPY --chown=airflow:root . /opt/demand-intel
RUN pip install --no-cache-dir "/opt/demand-intel[ml]"
ENV DI_DATA_DIR=/opt/demand-intel/data DI_EVALUATION_DIR=/opt/demand-intel/evaluation
