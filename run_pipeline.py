import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Preparación de datos", "src/prepare.py"),
    ("Validación de datos", "src/validate_data.py"),
    ("Entrenamiento del modelo", "src/train.py"),
]


def run_step(number, total, name, script):
    print(f"\n[{number}/{total}] {name} ({script})")
    result = subprocess.run([sys.executable, script], cwd=ROOT)
    if result.returncode != 0:
        print(f"\nPipeline DETENIDO: falló el paso '{name}' (código de salida {result.returncode})")
        sys.exit(result.returncode)


def main():
    start = time.time()
    for number, (name, script) in enumerate(STEPS, start=1):
        run_step(number, len(STEPS), name, script)
    print(f"\nPipeline completado con éxito en {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()