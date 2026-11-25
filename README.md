# Pipeline único y control de calidad de datos: clasificador de especies Iris

Seminario de Grado II, Especialización en Ciencia de Datos, Universidad Santo Tomás, Seccional Tunja.
Actividad 2.1 "Estandarización de Pipelines y Control de Calidad de Datos".

Autora: Laura Tatiana Rincón López

## Contenido

1. [Introducción](#1-introducción)
2. [Inicio rápido](#2-inicio-rápido)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Metodología](#4-metodología)
5. [Validación de datos de entrada](#5-validación-de-datos-de-entrada)
6. [Contrato de datos](#6-contrato-de-datos)
7. [Resultados](#7-resultados)
8. [Reproducción paso a paso desde cero](#8-reproducción-paso-a-paso-desde-cero)
9. [Solución de problemas](#9-solución-de-problemas)
10. [Limitaciones](#10-limitaciones)
11. [Uso de inteligencia artificial](#11-uso-de-inteligencia-artificial)
12. [Referencias](#12-referencias)

## 1. Introducción

En la Actividad 1.1 se construyó una línea base reproducible para clasificar la especie de una flor Iris
(setosa, versicolor o virginica) a partir de cuatro medidas en centímetros, con un Random Forest y registro
de experimentos en MLflow. Ese trabajo se ejecutaba celda por celda en un notebook, sin comprobar la
calidad de los datos antes de entrenar.

Este repositorio convierte ese experimento en un **pipeline único**: una secuencia de etapas que va desde la
extracción de los datos hasta la validación del modelo y que se ejecuta con un solo comando. Además incorpora
tres controles de calidad:

- **Validación de datos de entrada** escrita solo con la librería estándar de Python, que verifica tipos,
  nulos y rangos lógicos antes de entrenar, además de reglas entre archivos que detectan especies faltantes
  y fuga de datos entre entrenamiento y prueba.
- **Contrato de datos** en JSON que define qué debe recibir el modelo en la futura inferencia, con un ejemplo
  de entrada válida y uno de entrada inválida.
- **Umbrales de calidad del modelo**: si el modelo no alcanza los mínimos definidos, el pipeline termina con error.

## 2. Inicio rápido

Requisitos: Python 3.12 o 3.13 y Git. En Windows, desde la terminal de VS Code o Antigravity (PowerShell):

```bash
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>
python -m venv .venv                       # 1. Crea el entorno (una vez)
.venv\Scripts\activate                     # 2. Lo activa (cada vez que se abre la terminal)
python -m pip install -r requirements.txt  # 3. Instala las librerías (una vez)
python run_pipeline.py                     # 4. Ejecuta el pipeline
```

En macOS o Linux el paso 2 es `source .venv/bin/activate`.

El último comando ejecuta todo el pipeline. Si se dispone de `make` (Linux, macOS, WSL o Git Bash con make
instalado), el equivalente es `make pipeline`.

| Tarea | Python (cualquier sistema) | Makefile |
|---|---|---|
| Pipeline completo | `python run_pipeline.py` | `make pipeline` |
| Pipeline desde cero (borra resultados previos) | `python run_pipeline.py --reiniciar` | `make reproducir` |
| Solo validar datos | `python -m src.validate_data` | `make validate` |
| Validar los ejemplos del contrato | `python -m src.validate_data --ejemplos` | |
| Predecir con el ejemplo válido o inválido | `python -m src.predict --ejemplo valida` | `make predict` |
| Validar una solicitud propia | `python -m src.validate_data --entrada solicitud.json` | |
| Pruebas automatizadas | `python -m pytest` | `make test` |
| Juego "¡Rompe el pipeline!" (escenario 1 a 7) | `python romper_datos.py 3` y luego `python run_pipeline.py --sin-preparar` | `make romper E=3` |
| Ver los experimentos | `mlflow ui --backend-store-uri sqlite:///mlflow.db` | `make ui` |
| Ver todas las opciones | | `make help` |

## 3. Estructura del repositorio

```
.
├── Entrega2_LauraRincon.ipynb     Notebook comentado que recorre y ejecuta todo el pipeline
├── run_pipeline.py                Orquestador: ejecuta las 5 etapas con un solo comando
├── romper_datos.py                Sabotajes del juego de la Sesión 4 para probar los controles
├── Makefile                       Misma automatización con make
├── requirements.txt               Dependencias con versiones fijas
├── pytest.ini                     Configuración de las pruebas
├── config/
│   └── params.json                Semillas, partición, hiperparámetros, umbrales y MLflow
├── contracts/
│   └── data_contract.json         Contrato oficial de datos para la inferencia
├── src/
│   ├── utils.py                   Rutas, lectura de configuración, log y hash de archivos
│   ├── prepare.py                 Etapa 1: extracción y partición 80/20
│   ├── validate_data.py           Etapa 2: validación de datos (solo librería estándar)
│   ├── train.py                   Etapa 3: entrenamiento y registro en MLflow
│   ├── evaluate.py                Etapa 4: validación del modelo con umbrales de calidad
│   └── predict.py                 Inferencia que aplica el contrato antes de predecir
└── tests/
    ├── test_validate.py           Pruebas unitarias del validador
    └── test_model.py              Pruebas sobre datos procesados, modelo e inferencia
```

Carpetas que crea el pipeline y que no se suben a GitHub (ver `.gitignore`), porque se regeneran con
`python run_pipeline.py`:

| Carpeta o archivo | Contenido |
|---|---|
| `data/raw/iris.csv` | Dataset completo (150 registros) |
| `data/processed/` | `train.csv` (120) y `test.csv` (30) |
| `models/` | `model.pkl` (modelo seleccionado), `baseline.pkl` y `run_info.json` con los Run ID |
| `reports/` | `validation_report.json` y `evaluation_report.json` |
| `logs/pipeline.log` | Registro con fecha y hora de cada etapa, validaciones aprobadas y fallidas |
| `mlflow.db`, `mlruns/` | Base de datos y artefactos de MLflow |

## 4. Metodología

### 4.1 Datos

- **Fuente:** dataset Iris (Fisher, 1936), incluido en scikit-learn (`sklearn.datasets.load_iris`). Es
  público, no contiene datos personales y no requiere consentimiento ni anonimización.
- **Naturaleza:** datos estructurados, 150 registros, 4 variables numéricas continuas en centímetros y una
  variable objetivo categórica con 3 clases balanceadas (50 flores por especie).
- **Variables:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (características) y
  `species` (0 = setosa, 1 = versicolor, 2 = virginica). Los nombres originales de scikit-learn se
  renombran para que no tengan espacios ni unidades y coincidan con los de la API de la Semana 3.
- **Partición:** 80/20 estratificada por especie con `random_state=42`: 120 flores de entrenamiento (40 de
  cada especie) y 30 de prueba (10 de cada especie). En la Actividad 1.1 la partición no era estratificada y
  la prueba quedó con 10, 9 y 11 flores; estratificar garantiza que las tres especies tengan el mismo peso en
  la evaluación, como se propone en la Sesión 4. Por eso las 30 flores de prueba no son las mismas de la
  Actividad 1.1 y los resultados cambian (sección 7).

### 4.2 Etapas del pipeline

| Etapa | Script | Entrada | Salida | Detiene el pipeline si... |
|---|---|---|---|---|
| 1. prepare | `src/prepare.py` | scikit-learn | `data/raw`, `data/processed` | ocurre un error de lectura o escritura |
| 2. validate | `src/validate_data.py` | CSV + contrato | `reports/validation_report.json` | algún registro o regla de conjunto falla |
| 3. train | `src/train.py` | `train.csv` | `models/`, runs en MLflow | la validación no está aprobada o `train.csv` cambió |
| 4. evaluate | `src/evaluate.py` | `test.csv` + modelos | `reports/evaluation_report.json` | el modelo no alcanza un umbral |
| 5. test | `tests/` | todo lo anterior | resumen de pytest | alguna prueba falla |

Cada etapa se ejecuta en un proceso independiente y termina con código 0 si es correcta o 1 si falla. Tanto
`run_pipeline.py` como el `Makefile` se detienen en la primera etapa que falla, de modo que nunca se entrena
con datos rechazados.

### 4.3 Modelado

Se entrenan dos modelos `RandomForestClassifier` (scikit-learn) con `random_state=42`:

| Rol | Run en MLflow | n_estimators | max_depth | Justificación |
|---|---|---:|---:|---|
| Línea base | `Pipeline_Baseline` | 100 | None | Configuración por defecto, punto de referencia |
| Modelo seleccionado | `Pipeline_Run_2_n120_depth4` | 120 | 4 | Elegido en la Actividad 1.1 por equilibrio entre desempeño y simplicidad (árboles menos profundos) |

Los hiperparámetros se leen de `config/params.json`, por lo que se puede probar otra configuración sin
modificar el código.

### 4.4 Validación del modelo

- **Validación cruzada de 5 folds** sobre los 120 registros de entrenamiento. A diferencia de la Actividad
  1.1, donde se hizo con los 150 registros, aquí se excluye el conjunto de prueba para que ninguna métrica
  de entrenamiento lo use.
- **Métricas sobre prueba:** accuracy, F1 macro, número de errores, matriz de confusión y brecha entre
  accuracy de entrenamiento y de prueba.
- **Umbrales mínimos** (`config/params.json`): accuracy ≥ 0.90, F1 macro ≥ 0.90 y accuracy media de
  validación cruzada ≥ 0.90. Además, el modelo seleccionado no puede ser inferior a la baseline, el ejemplo
  válido del contrato debe recibir una predicción y el ejemplo inválido debe ser rechazado.
- El umbral de 0.90 es más exigente que el 0.80 usado en la demostración de la Sesión 4. La razón se ve en
  el juego de la sección 7.3: con 40 % de etiquetas erradas el modelo todavía alcanza 0.867, que pasaría
  un umbral de 0.80 pero no uno de 0.90.

### 4.5 Reproducibilidad y trazabilidad

| Elemento | Dónde queda registrado |
|---|---|
| Semillas (`split_seed=42`, `model_seed=42`) y partición estratificada | `config/params.json` y parámetros de cada run en MLflow |
| Versiones de Python, scikit-learn y MLflow | `requirements.txt` y etiquetas de cada run |
| Huella SHA-256 de `train.csv` | Reporte de validación y parámetro `train_sha256` en MLflow |
| Contrato, configuración y reporte de validación | Artefactos de cada run en MLflow |
| Resultado de la validación del modelo | Etiqueta `validacion_modelo` y artefacto `evaluacion/` del run |
| Ejecución de cada etapa | `logs/pipeline.log` |

El hash SHA-256 funciona como huella digital del archivo: si `train.csv` se modifica después de validarlo,
`train.py` lo detecta y se niega a entrenar.

Versiones fijadas en `requirements.txt`: scikit-learn 1.9.1, MLflow 3.18.0 (backend SQLite), pandas 3.0.5,
joblib 1.6.0 y pytest 9.1.1, probadas con Python 3.12 y 3.13.

## 5. Validación de datos de entrada

`src/validate_data.py` usa solo módulos de la librería estándar (`csv`, `json`, `math`, `argparse`). Las reglas
no están escritas en el código: se leen del contrato de datos, de modo que entrenamiento e inferencia se
validan contra la misma definición.

| Regla | Qué verifica | Ejemplo que la incumple |
|---|---|---|
| columnas | Están todas las columnas requeridas y no hay columnas de más | falta `petal_width`, llega `color` |
| nulo | Ningún valor vacío o faltante: `None`, `""`, `NaN`, `"NA"`, `"null"`, entre otros | `"sepal_width": null` |
| tipo | Características numéricas finitas y variable objetivo entera. En JSON un número entre comillas o un booleano se rechaza | `"5.1"`, `true`, `"abc"` |
| rango | Cada medida está entre su mínimo y máximo lógicos | `petal_length = -1.4` |
| dominio | La variable objetivo solo toma 0, 1 o 2 | `species = 4` |
| relacion | La longitud del pétalo es menor que la del sépalo (se cumple en las 150 flores) | columnas intercambiadas |

Reglas de conjunto, que comparan los archivos entre sí y que una fila aislada no puede revelar:

| Regla | Qué verifica | Ejemplo que la incumple |
|---|---|---|
| clases_completas | `train.csv` y `test.csv` contienen las tres especies | la extracción no trajo flores virginica |
| sin_solapamiento_train_test | Ninguna flor de prueba está también en entrenamiento (fuga de datos) | las 30 flores de prueba copiadas en `train.csv` |

La segunda regla responde a la lección principal de la Sesión 4: un pipeline solo detecta lo que alguien
le enseñó a revisar. Sin ella, copiar la prueba en el entrenamiento produce un accuracy perfecto y todas las
pruebas aprobadas. La regla tolera únicamente la flor que ya viene repetida en el dataset original.

Rangos lógicos definidos en el contrato. Corresponden al rango observado en las 150 flores, ampliado al
múltiplo de 0,5 cm más cercano y sin bajar de 0,1 cm, porque una medida física debe ser positiva:

| Variable | Observado | Rango lógico (cm) |
|---|---|---|
| sepal_length | 4.3 a 7.9 | 4.0 a 8.0 |
| sepal_width | 2.0 a 4.4 | 2.0 a 4.5 |
| petal_length | 1.0 a 6.9 | 1.0 a 7.0 |
| petal_width | 0.1 a 2.5 | 0.1 a 2.5 |

Además, a nivel de archivo se verifica que tenga registros y se reportan los registros duplicados como
advertencia. Iris tiene una flor repetida (filas 103 y 144 de `data/raw/iris.csv`, contando el encabezado
como fila 1); es un dato real, por eso no bloquea el pipeline, pero queda documentada en el log.

Ejemplo del log cuando un archivo tiene errores:

```
RECHAZADO | data/processed/train.csv | 120 registros | 4 errores {'nulo': 1, 'tipo': 1, 'rango': 1, 'dominio': 1}
  fila 4 | sepal_length | nulo | Valor nulo o vacío | valor=''
  fila 6 | petal_length | tipo | Se esperaba float y llegó str | valor='abc'
  fila 8 | petal_width | rango | Fuera del rango lógico [0.1, 2.5] cm | valor='9.5'
  fila 10 | species | dominio | Clase fuera de [0, 1, 2] | valor='4'
```

## 6. Contrato de datos

`contracts/data_contract.json` es el contrato oficial para la inferencia. Define:

- El modelo al que aplica y el orden de las características.
- Cada campo de entrada: tipo, unidad, si es requerido, si admite nulos, mínimo, máximo y descripción.
- La regla entre campos, las reglas de conjunto y el criterio con que se fijaron los rangos.
- La variable objetivo, que **no** debe enviarse al predecir: recibirla sería una fuga de datos.
- El formato de salida (`prediction` y `class_id`) y el manejo de errores.
- Un ejemplo de entrada válida, su salida esperada, un ejemplo de entrada inválida y los errores que debe
  producir.

Entrada válida y su salida:

```json
{"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.4}
```
```json
{"prediction": "versicolor", "class_id": 1}
```

Entrada inválida, con un incumplimiento de cada tipo:

```json
{"sepal_length": "5.1", "sepal_width": null, "petal_length": -1.4, "petal_width": 0.2, "species": 0}
```

| Campo | Regla | Motivo |
|---|---|---|
| sepal_length | tipo | Llega como texto y el contrato exige un número |
| sepal_width | nulo | El campo no admite nulos |
| petal_length | rango | -1.4 cm está por debajo del mínimo de 1.0 cm |
| species | columnas | Es la variable objetivo: enviarla en la inferencia sería fuga de datos |

`src/predict.py` valida cada registro contra el contrato antes de pasarlo al modelo y devuelve los errores
en lugar de una predicción cuando el registro no cumple. Así el modelo solo recibe datos con las mismas
columnas, tipos y rangos con los que fue entrenado, lo que evita el sesgo entre entrenamiento y servicio
(*training-serving skew*). La etapa de validación comprueba en cada ejecución que el ejemplo válido pase y
que el inválido produzca exactamente los cuatro errores anunciados.

## 7. Resultados

### 7.1 Desempeño

Tabla 1. Desempeño de la línea base y del modelo seleccionado (semilla 42, partición estratificada, 30 flores
de prueba).

| Modelo | Accuracy entrenamiento | CV 5 folds (media ± desv.) | Accuracy prueba | F1 macro prueba | Errores en prueba |
|---|---:|---:|---:|---:|---:|
| Pipeline_Baseline | 1.000 | 0.950 ± 0.017 | 0.900 | 0.900 | 3 de 30 |
| Pipeline_Run_2_n120_depth4 | 1.000 | 0.950 ± 0.017 | 0.967 | 0.967 | 1 de 30 |

Ambos modelos tienen la misma validación cruzada sobre el entrenamiento. En prueba, el modelo seleccionado
falla en 1 flor y la baseline en 3. La diferencia equivale a 2 flores, por lo que es modesta, pero
confirma que limitar la profundidad a 4 no reduce el desempeño. A diferencia de la Actividad 1.1, con esta
partición el modelo seleccionado también acierta todo el entrenamiento (1.000), así que el argumento de
"no memorizar" ya no lo distingue de la baseline; la selección se sostiene en el desempeño en prueba y en
la simplicidad.

### 7.2 Controles de calidad

Tabla 2. Controles de calidad del pipeline.

| Control | Resultado |
|---|---|
| Validación de datos (raw, train, test) | Aprobada, 0 errores, 1 advertencia por la flor duplicada |
| Reglas de conjunto | Aprobadas: tres especies en cada archivo y sin solapamiento |
| Coherencia del contrato con el validador | Aprobada |
| Umbrales del modelo seleccionado | Aprobados |
| Ejemplo válido del contrato | Predicho como versicolor |
| Ejemplo inválido del contrato | Rechazado con 4 errores |
| Pruebas automatizadas | 38 aprobadas |

Al ejecutar el pipeline dos veces desde cero se obtienen las mismas métricas; solo cambian los Run ID y el
tiempo de entrenamiento.

### 7.3 Juego "¡Rompe el pipeline!" de la Sesión 4

`romper_datos.py` reproduce los siete sabotajes de la clase sobre `train.csv`. Después se ejecuta
`python run_pipeline.py --sin-preparar` (o `make romper E=<n>`) para ver qué control lo detiene.

Tabla 3. Etapa que detiene cada sabotaje en la clase y en este repositorio.

| Sabotaje | En la clase | En este repositorio | Evidencia en el log |
|---|---|---|---|
| 1. «cinco» en lugar de un número | Validación | Validación | `tipo`: se esperaba float y llegó str |
| 2. Celda vacía en `petal_width` | Validación | Validación | `nulo`: valor nulo o vacío |
| 3. Registro en milímetros | Validación | Validación | `rango`: 4 errores, uno por medida |
| 4. Especie con código 3 | Validación | Validación | `dominio`: clase fuera de [0, 1, 2] |
| 5. Sin flores virginica | Pruebas | Validación | `clases_completas`: faltan especies |
| 6. 40 % de etiquetas erradas | Pruebas | Validación del modelo | accuracy 0.867, F1 y CV bajo el umbral de 0.90 |
| 7. Prueba copiada en entrenamiento | Nadie | Validación | `sin_solapamiento_train_test`: 30 de 30 flores repetidas |

Los sabotajes 5 y 7 se detienen ahora antes de entrenar, gracias a las reglas de conjunto. El 6 no se puede
detectar mirando los datos, porque las etiquetas erradas siguen siendo valores válidos; lo detiene la
validación del modelo. Para volver a los datos correctos basta con ejecutar `python run_pipeline.py`.

## 8. Reproducción paso a paso desde cero

1. Instalar Python 3.12 o 3.13 y Git.
2. Clonar el repositorio y entrar a la carpeta (sección 2).
3. Crear y activar un entorno virtual e instalar las dependencias con
   `python -m pip install -r requirements.txt`.
4. Ejecutar `python run_pipeline.py --reiniciar`. La opción `--reiniciar` borra `mlflow.db`, `mlruns`, `data`,
   `models`, `reports` y `logs` para partir de cero. El pipeline tarda menos de un minuto y termina con el
   mensaje `Pipeline completado`.
5. Revisar `logs/pipeline.log`, `reports/validation_report.json` y `reports/evaluation_report.json`.
6. Abrir MLflow con `mlflow ui --backend-store-uri sqlite:///mlflow.db` e ir a
   `http://127.0.0.1:5000` → Experiments → `Proyecto_Final_Grado`. Cada run contiene parámetros, métricas,
   etiquetas, el modelo con su firma de entrada y los artefactos `contrato/`, `validacion/`, `config/` y
   `requirements.txt`. El run del modelo seleccionado incluye además `evaluacion/`.
7. Probar el contrato con `python -m src.predict --ejemplo valida` y `python -m src.predict --ejemplo invalida`.

Para comprobar que el control de calidad funciona, se puede editar `data/processed/train.csv` (por ejemplo,
borrar un valor o escribir `abc`) y ejecutar `python -m src.validate_data` seguido de `python -m src.train`: la
validación sale con código 1 y el entrenamiento se niega a continuar.

El notebook `Entrega2_LauraRincon.ipynb` recorre los mismos pasos con explicaciones. Debe abrirse desde la
raíz del repositorio, al lado de `run_pipeline.py`, porque toma esa carpeta como punto de partida.

## 9. Solución de problemas

| Situación | Solución |
|---|---|
| `make` no se reconoce en Windows | Usar `python run_pipeline.py`, que hace lo mismo |
| `ModuleNotFoundError: No module named 'src'` | Ejecutar los comandos desde la raíz del repositorio y con `python -m` |
| `--reiniciar` no puede borrar `mlflow.db` | Cerrar MLflow UI, que mantiene el archivo en uso |
| El puerto 5000 está ocupado | `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001` |
| PowerShell no deja activar el entorno (`running scripts is disabled`) | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` y volver a activar |
| El pipeline se detiene en validate después del juego | Es lo esperado: `python run_pipeline.py` regenera los datos limpios |

## 10. Limitaciones

- Iris es un dataset pequeño y fácil de separar. Con 30 flores de prueba cada error vale 3.3 puntos
  porcentuales, y el resultado depende de qué flores quedan en ese conjunto: en la Actividad 1.1, con 10
  semillas distintas, la accuracy varió entre 0.83 y 1.0. Los umbrales de 0.90 son un control mínimo, no una
  garantía de desempeño.
- Un pipeline solo detecta lo que se le enseñó a revisar. Por ejemplo, un porcentaje de etiquetas erradas
  menor al 40 % podría no bajar la accuracy del umbral y pasar todos los controles.
- Los rangos lógicos se derivan del propio dataset. Con datos reales deberían definirse con expertos del
  dominio y revisarse cuando cambie la población de origen.
- La validación no detecta cambios de distribución (*data drift*) entre los datos de entrenamiento y los de
  producción, que sería el siguiente paso de control de calidad.

## 11. Uso de inteligencia artificial

Conforme a la ruta de aprendizaje, se declara el uso de [herramienta y versión] como apoyo en [describir:
por ejemplo, estructuración del código, revisión de errores, sugerencias de redacción del README]. El
diseño del experimento, las decisiones metodológicas, la verificación de los resultados y la redacción
final son responsabilidad de la autora.

## 12. Referencias

Contreras Ortiz, M. S., Puerto Moreno, S. A., Suárez Rojas, A. L., & Vela Beltrán, D. A. (2026). *Seminario
de Grado II*. Ediciones USTA Tunja.

Contreras Ortiz, M. S. (2026, 19 de noviembre). *Estandarización de pipelines y control de calidad de datos*
[Presentación de la Sesión 4]. Seminario de Grado II, Universidad Santo Tomás, Seccional Tunja.

Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics, 7*(2),
179–188. https://doi.org/10.1111/j.1469-1809.1936.tb02137.x

MLOps Open Source - Santoto. (2026). *MLOps código abierto: laboratorio práctico*.
https://dialvebe.github.io/mlops-course-santoto/

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer,
P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., &
Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*,
2825–2830.

Zaharia, M., Chen, A., Davidson, A., Ghodsi, A., Hong, S. A., Konwinski, A., Murching, S., Nykodym, T.,
Ogilvie, P., Parkhe, M., Xie, F., & Zumar, C. (2018). Accelerating the machine learning lifecycle with
MLflow. *IEEE Data Engineering Bulletin, 41*(4), 39–45.
