# ESTUDUANTE: Erika Lorena Guzman Gonzalez
# Dataset: Iris
# Version Python 3.12.8

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Proyecto_Final_Grado")

with mlflow.start_run(run_name="Baseline"):
    n_estimators = 100  
    max_depth = None    

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=24
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, name="iris_model",
                             skops_trusted_types=["sklearn.tree._tree.Tree"])
    print(f"Modelo registrado con accuracy: {accuracy}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")