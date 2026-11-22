import csv
import json
import sys
from pathlib import Path

# Raíz del repositorio (este archivo está en src/)
ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT / "data_contract.json"
DEFAULT_FILES = ["data/processed/train.csv", "data/processed/test.csv"]

# Textos que se consideran valores nulos (se comparan en minúsculas)
NULL_VALUES = {"", "nan", "none", "null", "na"}

# Tipos que puede declarar el contrato
TYPES = {"float": float, "int": int}


def load_contract(path=CONTRACT_PATH):
    """Lee el contrato de datos (JSON) con la biblioteca estándar."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_schema(contract):
    """Construye las reglas por columna a partir del contrato.

    El orden es el de feature_order, seguido de la variable objetivo,
    que es el orden de las columnas de los CSV.
    """
    fields = contract["input"]["fields"]
    schema = {}
    for name in contract["input"]["feature_order"]:
        rules = fields[name]
        schema[name] = {
            "type": TYPES[rules["type"]],
            "nullable": rules.get("nullable", False),
            "min": rules["min"],
            "max": rules["max"],
        }

    output = contract["output"]
    output_type = TYPES[output["type"]]
    schema[output["field"]] = {
        "type": output_type,
        "nullable": False,
        "allowed": {output_type(value) for value in output["allowed_values"]},
    }
    return schema


CONTRACT = load_contract()
FEATURES = CONTRACT["input"]["feature_order"]
SCHEMA = build_schema(CONTRACT)


def validate_value(column, raw, rules):
    """Devuelve un mensaje de error si el valor no cumple las reglas, o None si es válido."""
    text = (raw or "").strip()

    # 1. Ausencia de nulos
    if text.lower() in NULL_VALUES:
        if rules.get("nullable", False):
            return None
        return f"valor nulo o vacío en '{column}'"

    # 2. Tipo correcto
    try:
        value = rules["type"](text)
    except ValueError:
        return f"'{column}' debe ser {rules['type'].__name__}, se recibió '{text}'"

    # 3. Rango lógico o valores permitidos
    if "allowed" in rules and value not in rules["allowed"]:
        return f"'{column}' = {value} no está entre los valores permitidos {sorted(rules['allowed'])}"
    if "min" in rules and not (rules["min"] <= value <= rules["max"]):
        return f"'{column}' = {value} fuera del rango [{rules['min']}, {rules['max']}]"

    return None


def validate_record(record):
    """Valida una entrada individual (diccionario) contra el contrato.

    Es la validación que se aplicaría a una entrada de inferencia.
    Devuelve la lista de errores (vacía si la entrada es válida).
    """
    errors = []

    missing = [name for name in FEATURES if name not in record]
    extra = [name for name in record if name not in FEATURES]
    if missing:
        errors.append(f"faltan campos obligatorios: {missing}")
    if extra:
        errors.append(f"campos no permitidos: {extra}")

    for name in FEATURES:
        if name in record:
            value = record[name]
            raw = "" if value is None else str(value)
            problem = validate_value(name, raw, SCHEMA[name])
            if problem:
                errors.append(problem)
    return errors


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