import csv
import sys
from pathlib import Path

# Textos que podrían considerarse como valores nulos
NULL_VALUES = {"", "nan", "none", "null", "na"}

# Reglas para cada columna
SCHEMA = {
    "sepal length (cm)": {"type": float, "min": 3.0, "max": 9.0},
    "sepal width (cm)": {"type": float, "min": 1.0, "max": 6.0},
    "petal length (cm)": {"type": float, "min": 0.5, "max": 8.0},
    "petal width (cm)": {"type": float, "min": 0.1, "max": 3.5},
    "species": {"type": int, "allowed": {0, 1, 2}},
}

DEFAULT_FILES = ["data/processed/train.csv", "data/processed/test.csv"]


def validate_value(column, raw, rules):
    """Devuelve un mensaje de error si el valor no cumple las reglas, o None si es válido."""
    text = (raw or "").strip()

    # Para ausencia de nulos
    if text.lower() in NULL_VALUES:
        return f"valor nulo o vacío en '{column}'"

    # Para el tipo correcto recibido
    try:
        value = rules["type"](text)
    except ValueError:
        return f"'{column}' debe ser {rules['type'].__name__}, se recibió '{text}'"

    # Valores permitidos
    if "allowed" in rules and value not in rules["allowed"]:
        return f"'{column}' = {value} no está entre los valores permitidos {sorted(rules['allowed'])}"
    if "min" in rules and not (rules["min"] <= value <= rules["max"]):
        return f"'{column}' = {value} fuera del rango [{rules['min']}, {rules['max']}]"

    return None


def validate_file(path):
    """Valida un CSV completo y devuelve la lista de errores encontrados."""
    path = Path(path)
    if not path.exists():
        return [f"{path}: el archivo no existe"]

    errors = []
    row_count = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != list(SCHEMA):
            return [f"{path}: columnas inesperadas {reader.fieldnames}, se esperaban {list(SCHEMA)}"]

        for line_number, row in enumerate(reader, start=2):
            row_count += 1
            if None in row:
                errors.append(f"{path} fila {line_number}: tiene más columnas de las esperadas")
                continue
            for column, rules in SCHEMA.items():
                problem = validate_value(column, row[column], rules)
                if problem:
                    errors.append(f"{path} fila {line_number}: {problem}")

    if row_count == 0:
        errors.append(f"{path}: el archivo no tiene filas de datos")
    return errors


def main():
    files = sys.argv[1:] or DEFAULT_FILES
    all_errors = []
    for file in files:
        all_errors.extend(validate_file(file))

    if all_errors:
        print(f"Validación FALLIDA: {len(all_errors)} problema(s) encontrado(s)")
        for error in all_errors:
            print(f" - {error}")
        sys.exit(1)

    print(f"Validación exitosa: {len(files)} archivo(s) cumplen tipos, nulos y rangos")


if __name__ == "__main__":
    main()