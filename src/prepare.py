import csv
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path("data/processed")
TEST_SIZE = 0.2
RANDOM_STATE = 24


def save_csv(path, header, features, labels):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row, label in zip(features, labels):
            writer.writerow([float(v) for v in row] + [int(label)])


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    header = list(iris.feature_names) + ["species"]
    save_csv(OUTPUT_DIR / "train.csv", header, X_train, y_train)
    save_csv(OUTPUT_DIR / "test.csv", header, X_test, y_test)

    print(f"Datos preparados: {len(X_train)} filas de entrenamiento y {len(X_test)} de prueba")


if __name__ == "__main__":
    main()