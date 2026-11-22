# EstandarizacionPipeline_ControlCalidadDatos

Pipeline reproducible de Machine Learning sobre el dataset **Iris**: prepara los datos, los valida con reglas de calidad (sin librerías externas), entrena un modelo `RandomForestClassifier` y registra el experimento en **MLflow**. Todo se ejecuta con **un solo comando**.

- **Autora:** Erika Lorena Guzman Gonzalez
- **Contexto:** Seminario de Grado II, Especialización en Ciencia de Datos (Universidad Santo Tomás)
- **Repositorio:** https://github.com/Erika-GGz/EstandarizacionPipeline_ControlCalidadDatos

---

## 1. Qué hace el pipeline

El pipeline se orquesta con `run_pipeline.py` y ejecuta tres etapas en orden. Si una etapa falla, el pipeline se detiene y no ejecuta las siguientes.

| Etapa | Script | Qué hace |
|---|---|---|
| 1. Preparación | `src/prepare.py` | Carga Iris, la divide en entrenamiento (80 %) y prueba (20 %) y guarda `data/processed/train.csv` y `data/processed/test.csv`. |
| 2. Validación | `src/validate_data.py` | Verifica tipos, ausencia de nulos y rangos lógicos en ambos CSV. Si hay errores, termina con código de salida 1 y **el entrenamiento no se ejecuta**. |
| 3. Entrenamiento | `src/train.py` | Entrena el modelo, calcula el accuracy y registra parámetros, métrica y modelo en MLflow. |

---

## 2. Requisitos previos

- **Python 3.12** (desarrollado y probado con 3.12.8).
- **Git**, para clonar el repositorio.
- Sistema operativo: probado en **Windows 11 con PowerShell**. En macOS o Linux los pasos son equivalentes (cambia solo el comando de activación del entorno virtual, ver sección 4).

No se requiere `make` ni ninguna otra herramienta: la orquestación está escrita en Python.

---

## 3. Estructura del repositorio

```
EstandarizacionPipeline_ControlCalidadDatos/
├── src/
│   ├── prepare.py            # Etapa 1: preparación de datos
│   ├── validate_data.py      # Etapa 2: validación de datos (solo biblioteca estándar)
│   └── train.py              # Etapa 3: entrenamiento y registro en MLflow
├── test/
│   ├── test_model.py         # Pruebas automáticas (pytest)
│   └── data/
│       └── train_errores.csv # Datos con errores a propósito para probar la validación
├── data_contract.json        # Contrato de datos: única fuente de reglas de validación
├── run_pipeline.py           # Script de orquestación (comando único)
├── pytest.ini                # Configuración de pytest
├── requirements.txt          # Dependencias
├── .gitignore
└── README.md
```

Los siguientes elementos **se generan al ejecutar el pipeline** y no se versionan en Git:

- `data/processed/`: CSV de entrenamiento y prueba.
- `mlflow.db` y `mlruns/`: historial local de experimentos de MLflow.
- `venv/`: entorno virtual.

---

## 4. Instalación desde cero

### 4.1 Clonar el repositorio

```
git clone https://github.com/Erika-GGz/EstandarizacionPipeline_ControlCalidadDatos.git
cd EstandarizacionPipeline_ControlCalidadDatos
```

### 4.2 Crear el entorno virtual

```
python -m venv venv
```

### 4.3 Activar el entorno virtual

**Windows (PowerShell):**

```
.\venv\Scripts\Activate.ps1
```

Si PowerShell indica que la ejecución de scripts está deshabilitada, ejecuta una vez:

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**macOS / Linux:**

```
source venv/bin/activate
```

Cuando el entorno está activo, el prompt muestra `(venv)` al inicio.

### 4.4 Instalar las dependencias

```
pip install -r requirements.txt
```

Las dependencias son `mlflow` y `scikit-learn` para el pipeline, y `pytest` para las pruebas automáticas.

---

## 5. Ejecutar el pipeline

Desde la **raíz del repositorio** y con el entorno virtual activo:

```
python run_pipeline.py
```

Salida esperada (el Run ID y el tiempo cambian en cada ejecución):

```
[1/3] Preparación de datos (src/prepare.py)
Datos preparados: 120 filas de entrenamiento y 30 de prueba

[2/3] Validación de datos (src/validate_data.py)
Validación exitosa: 2 archivo(s) cumplen tipos, nulos y rangos

[3/3] Entrenamiento del modelo (src/train.py)
Modelo registrado con accuracy: 1.0
Run ID: <identificador del run>

Pipeline completado con éxito en <segundos> s
```

Cada ejecución crea un run nuevo en MLflow, por lo que el historial crece con cada corrida.

También puedes ejecutar cada etapa por separado, siempre desde la raíz del repositorio:

```
python src/prepare.py
python src/validate_data.py
python src/train.py
```

---

## 6. Visualizar los resultados en MLflow

Con el entorno activo y desde la raíz del repositorio:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre en el navegador http://127.0.0.1:5000. En el experimento **Proyecto_Final_Grado** encontrarás los runs (nombre `Baseline`) con:

- **Parámetros:** `n_estimators` y `max_depth`.
- **Métrica:** `accuracy`.
- **Modelo:** registrado como `iris_model`.

---

## 7. Validación de datos

La validación está implementada **sin librerías externas** (solo `csv`, `json`, `sys` y `pathlib` de la biblioteca estándar de Python) en `src/validate_data.py`. Las reglas **se leen de `data_contract.json`**, por lo que el contrato y la validación nunca se contradicen. Revisa tres aspectos en cada fila de `train.csv` y `test.csv`:

1. **Tipos de datos correctos:** las medidas deben ser decimales y `species` un entero.
2. **Ausencia de valores nulos:** se consideran nulos las celdas vacías y los textos `nan`, `none`, `null` y `na`.
3. **Rangos lógicos:** cada variable debe estar dentro de un rango con sentido biológico.

Además comprueba que el archivo tenga exactamente las columnas esperadas (en el orden del contrato) y al menos una fila de datos. El mismo módulo ofrece `validate_record`, que aplica las reglas a una entrada individual, como la que recibiría el modelo en inferencia.

| Columna | Tipo | Regla |
|---|---|---|
| `sepal length (cm)` | decimal | entre 3.0 y 9.0 |
| `sepal width (cm)` | decimal | entre 1.0 y 6.0 |
| `petal length (cm)` | decimal | entre 0.5 y 8.0 |
| `petal width (cm)` | decimal | entre 0.1 y 3.5 |
| `species` | entero | solo 0, 1 o 2 |

Los límites dejan un margen sobre los valores observados en Iris (por ejemplo, el sépalo mide entre 4.3 y 7.9 cm de largo), de modo que una flor real algo más grande no sea rechazada, pero valores imposibles (negativos, cero o muy grandes) sí lo sean.

### Comportamiento ante errores

La validación recorre todo el archivo, reúne **todos** los errores (con archivo, fila, columna y motivo) y termina con código de salida **1**. Si todo es correcto, termina con código **0**. `run_pipeline.py` usa ese código para detener el pipeline antes de entrenar.

### Cómo comprobar que la validación funciona

El archivo `test/data/train_errores.csv` es una copia de los datos con cuatro defectos introducidos a propósito (un nulo, un texto, un valor fuera de rango y una especie no permitida):

```
python src/validate_data.py test/data/train_errores.csv
```

Salida esperada (en Windows las rutas usan `\`):

```
Validación FALLIDA: 4 problema(s) encontrado(s)
 - test\data\train_errores.csv fila 12: valor nulo o vacío en 'sepal length (cm)'
 - test\data\train_errores.csv fila 64: 'sepal width (cm)' debe ser float, se recibió 'abc'
 - test\data\train_errores.csv fila 97: 'petal width (cm)' = 50.0 fuera del rango [0.1, 3.5]
 - test\data\train_errores.csv fila 108: 'species' = 7 no está entre los valores permitidos [0, 1, 2]
```

En PowerShell puedes consultar el código de salida con `echo $LASTEXITCODE` (debe mostrar `1`). Con los datos buenos (`python src/validate_data.py`) debe mostrar `0`.

### Qué pasa si los datos del pipeline son inválidos

Si las reglas rechazan los datos reales, el pipeline se detiene en la etapa 2 y no entrena:

```
Pipeline DETENIDO: falló el paso 'Validación de datos' (código de salida 1)
```

---

## 8. Contrato de datos

El archivo `data_contract.json` es el contrato oficial de entrada para la futura inferencia del modelo. Contiene:

- **`model`:** el modelo al que aplica (nombre, algoritmo, tarea y variable objetivo).
- **`input`:** los 4 campos obligatorios, con su tipo, si admiten nulos y el rango permitido, y `feature_order`, el orden en que el modelo espera las variables (las recibe por posición, no por nombre).
- **`output`:** el campo `species` y sus valores (`0` setosa, `1` versicolor, `2` virginica).
- **`examples.valid_input`:** un ejemplo de entrada válida.
- **`examples.invalid_input`:** un ejemplo de entrada inválida, con las razones por las que se rechaza (valor fuera de rango, nulo y tipo incorrecto).

Para verificar que el archivo es un JSON válido:

```
python -m json.tool data_contract.json
```

El contrato es la **única fuente de verdad**: para cambiar un rango o un tipo basta con editar `data_contract.json`, y `src/validate_data.py` lo aplica sin tocar el código.

---

## 9. Parámetros fijos y reproducibilidad

| Elemento | Valor |
|---|---|
| Dataset | Iris (`sklearn.datasets.load_iris`), 150 muestras |
| División train/test | 80 % / 20 % (120 y 30 filas) |
| Semilla de la división | `random_state=24` |
| Modelo | `RandomForestClassifier` |
| `n_estimators` | 100 |
| `max_depth` | `None` |
| Semilla del modelo | `random_state=42` |
| Experimento MLflow | `Proyecto_Final_Grado` (run `Baseline`) |

Con estos valores fijos, el resultado esperado es un **accuracy de 1.0** sobre el conjunto de prueba.

---

## 10. Metodología

### 10.1 Datos y muestra

- **Fuente:** dataset Iris incluido en scikit-learn (`load_iris`), público y de uso didáctico. No requiere descargas externas.
- **Naturaleza de los datos:** estructurados (tabulares).
- **Tamaño total:** 150 muestras, sin valores faltantes.
- **Variables:** 4 variables numéricas continuas (longitud y ancho del sépalo y del pétalo, en centímetros) y una variable objetivo categórica codificada como entero (`species`: 0 setosa, 1 versicolor, 2 virginica). En el dataset completo, cada clase tiene 50 muestras.
- **Muestra utilizada:** se usa el dataset completo, dividido de forma aleatoria (no estratificada, para conservar la misma partición de la actividad anterior) en 120 muestras de entrenamiento (80 %) y 30 de prueba (20 %). La semilla `random_state=24` hace que la partición sea idéntica en cada ejecución.

### 10.2 Preprocesamiento

No se aplicaron transformaciones: no hay valores faltantes que imputar, la variable objetivo ya viene codificada, las clases están balanceadas y Random Forest no requiere estandarizar las variables. El control de calidad se realiza en la etapa de validación (sección 7), que impide entrenar con datos de tipo incorrecto, con nulos o fuera de rango.

### 10.3 Modelado

- **Algoritmo:** `RandomForestClassifier` (scikit-learn).
- **Hiperparámetros:** `n_estimators=100`, `max_depth=None`, `random_state=42`; el resto, con los valores por defecto de scikit-learn (ver sección 9).
- **Frameworks:** scikit-learn para el modelo y MLflow para registrar parámetros, métrica y modelo. Las dependencias están en `requirements.txt`.
- **Hardware:** ejecución en CPU de un equipo personal; no se requiere GPU.

### 10.4 Validación y evaluación

- **Esquema:** partición entrenamiento/prueba (hold-out), sin validación cruzada, porque el objetivo del proyecto es demostrar un pipeline reproducible y no seleccionar el mejor modelo.
- **Métrica:** *accuracy* (proporción de predicciones correctas), calculada sobre las 30 muestras de prueba, que el modelo no vio durante el entrenamiento.
- **Limitación:** con 30 muestras de prueba, cada error equivale a 3.3 puntos porcentuales, e Iris es un dataset sencillo y limpio. Un accuracy alto no es, por sí solo, evidencia de buen desempeño con datos reales.

---

## 11. Resultados

**Tabla 1.** Resultado del modelo baseline sobre el conjunto de prueba.

| Métrica | Valor | Conjunto de evaluación |
|---|---|---|
| Accuracy | 1.0 (30 de 30 predicciones correctas) | Prueba (30 muestras) |

Con las semillas fijas (sección 9), el accuracy fue 1.0 en todas las ejecuciones realizadas. El Run ID cambia en cada ejecución porque MLflow crea un run nuevo cada vez.

---

## 12. Pruebas automáticas

Las pruebas (`pytest`) están en `test/test_model.py` y su configuración en `pytest.ini`. Con el entorno activo, desde la raíz del repositorio:

```
python -m pytest
```

Qué verifican:

- Los archivos generados por `prepare.py` existen y tienen 120 y 30 filas.
- Los datos reales pasan la validación.
- `test/data/train_errores.csv` es rechazado con sus 4 errores, y el script de validación termina con código 1 (y con 0 si los datos son válidos).
- El contrato es coherente con los datos: mismas columnas y mismo orden.
- Los ejemplos del contrato se comportan como declaran: el válido pasa y el inválido se rechaza con sus 3 problemas.
- El accuracy del último run registrado en MLflow supera el mínimo de 0.8. Esta prueba se omite (no falla) si aún no se ha ejecutado el pipeline con `python run_pipeline.py`.

---

## 13. Notas y solución de problemas

- **Ejecuta siempre desde la raíz del repositorio.** Las rutas de los datos y de la base de MLflow (`mlflow.db`) son relativas a esa carpeta.
- **Aviso de GitPython (`Failed to import Git ...`):** MLflow intenta leer la versión del código con Git. Si Git no está instalado o no está en el PATH, muestra este mensaje, que es **inofensivo** y no afecta el resultado. Se elimina instalando Git y reiniciando la terminal, o definiendo `GIT_PYTHON_REFRESH=quiet`.
- **`(venv)` no aparece tras activar el entorno:** cierra la terminal, ábrela de nuevo en la carpeta del proyecto y repite la activación.
- **Decisiones de diseño:** la orquestación se hace con un script de Python (y no con un Makefile) para que funcione igual en Windows, macOS y Linux sin instalar herramientas adicionales; la validación usa solo la biblioteca estándar para cumplir el requisito de no depender de librerías externas.

---

## 14. Referencia

Guía del curso: *MLOps Open Source - Santoto* (2026), https://dialvebe.github.io/mlops-course-santoto/