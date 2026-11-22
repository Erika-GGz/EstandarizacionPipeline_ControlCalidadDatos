# ESTUDUANTE: Erika Lorena Guzman Gonzalez
# Dataset: Iris
# Version Python 3.12.8

import csv

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
TARGET = "species"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Proyecto_Final_Grado")


def load_split(path):
    """Lee un CSV preparado y devuelve (características, etiquetas)."""
    features, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(int(row[TARGET]))
            features.append([float(v) for k, v in row.items() if k != TARGET])
    return features, labels


with mlflow.start_run(run_name="Baseline"):
    n_estimators = 100
    max_depth = None

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    X_train, y_train = load_split(TRAIN_PATH)
    X_test, y_test = load_split(TEST_PATH)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(
        model,
        name="iris_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )
    print(f"Modelo registrado con accuracy: {accuracy}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")