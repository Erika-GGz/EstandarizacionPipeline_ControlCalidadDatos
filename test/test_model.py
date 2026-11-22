import csv
import subprocess
import sys
from pathlib import Path

import pytest

import validate_data  # se importa desde src/ gracias a pytest.ini

ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = Path(__file__).resolve().parent / "data"
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
MLFLOW_DB = ROOT / "mlflow.db"
EXPERIMENT = "Proyecto_Final_Grado"
MIN_ACCURACY = 0.8


def count_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


@pytest.fixture(scope="session")
def prepared_data():
    """Ejecuta prepare.py para que las pruebas no dependan de ejecuciones previas."""
    subprocess.run([sys.executable, "src/prepare.py"], cwd=ROOT, check=True)
    return TRAIN_CSV, TEST_CSV


# ---------- Datos preparados ----------

def test_prepared_files_exist(prepared_data):
    for path in prepared_data:
        assert path.exists(), f"No existe {path}"
        assert path.stat().st_size > 0, f"{path} está vacío"


def test_split_sizes(prepared_data):
    assert count_rows(TRAIN_CSV) == 120
    assert count_rows(TEST_CSV) == 30


# ---------- Validación de datos ----------

def test_real_data_passes_validation(prepared_data):
    for path in prepared_data:
        assert validate_data.validate_file(path) == []


def test_invalid_file_is_rejected():
    errors = validate_data.validate_file(TEST_DATA / "train_errores.csv")
    assert len(errors) == 4
    text = " ".join(errors)
    assert "nulo" in text
    assert "debe ser float" in text
    assert "fuera del rango" in text
    assert "no está entre los valores permitidos" in text


def test_validation_exit_codes(prepared_data):
    bad = subprocess.run(
        [sys.executable, "src/validate_data.py", str(TEST_DATA / "train_errores.csv")],
        cwd=ROOT,
        capture_output=True,
    )
    good = subprocess.run(
        [sys.executable, "src/validate_data.py"],
        cwd=ROOT,
        capture_output=True,
    )
    assert bad.returncode == 1
    assert good.returncode == 0


# ---------- Contrato de datos ----------

def test_contract_structure():
    contract = validate_data.load_contract()
    for key in ("contract_name", "version", "model", "input", "output", "examples"):
        assert key in contract, f"Falta '{key}' en el contrato"
    assert set(contract["input"]["fields"]) == set(contract["input"]["feature_order"])


def test_contract_matches_dataset_columns(prepared_data):
    contract = validate_data.load_contract()
    expected = contract["input"]["feature_order"] + [contract["output"]["field"]]
    with open(TRAIN_CSV, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header == expected


def test_contract_valid_example_passes():
    example = validate_data.load_contract()["examples"]["valid_input"]
    assert validate_data.validate_record(example) == []


def test_contract_invalid_example_is_rejected():
    example = validate_data.load_contract()["examples"]["invalid_input"]
    errors = validate_data.validate_record(example["input"])
    assert len(errors) == len(example["reasons"])


# ---------- Modelo ----------

def test_model_accuracy_minimum():
    if not MLFLOW_DB.exists():
        pytest.skip("No hay historial de MLflow: ejecuta primero python run_pipeline.py")

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{MLFLOW_DB.as_posix()}")
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        pytest.skip(f"No existe el experimento '{EXPERIMENT}': ejecuta primero python run_pipeline.py")

    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        pytest.skip("El experimento no tiene runs: ejecuta primero python run_pipeline.py")

    accuracy = runs[0].data.metrics["accuracy"]
    assert accuracy >= MIN_ACCURACY, f"accuracy {accuracy} por debajo del mínimo {MIN_ACCURACY}"