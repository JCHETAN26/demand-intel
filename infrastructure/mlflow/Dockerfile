FROM python:3.11-slim
# Keep the server on the same major version as the client in uv.lock (mlflow 3.x).
RUN pip install --no-cache-dir "mlflow>=3.1,<4" psycopg2-binary
EXPOSE 5000
