import argparse
import os
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss


def main(data_path: str):
    # Paksa tracking store ke root repo (biar konsisten dengan mlflow run)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or "file:../mlruns"
    mlflow.set_tracking_uri(tracking_uri)

    df = pd.read_csv(data_path)
    X = df.drop(columns=["income_label"])
    y = df["income_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # IMPORTANT:
    # Kalau script dijalankan via `mlflow run`, MLflow sudah bikin run dan set env MLFLOW_RUN_ID
    env_run_id = os.getenv("MLFLOW_RUN_ID")

    if env_run_id:
        mlflow.start_run(run_id=env_run_id)
    else:
        mlflow.start_run(run_name="ci_train_logreg")

    try:
        model = LogisticRegression(max_iter=2000, solver="liblinear")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        mlflow.log_metric("test_accuracy", float(accuracy_score(y_test, y_pred)))
        mlflow.log_metric("test_f1", float(f1_score(y_test, y_pred)))
        mlflow.log_metric("test_roc_auc", float(roc_auc_score(y_test, y_proba)))
        mlflow.log_metric("test_log_loss", float(log_loss(y_test, y_proba)))

        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        print(f"Run ID: {run_id}")
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="adult_preprocessing/processed.csv")
    args = p.parse_args()
    main(args.data_path)
