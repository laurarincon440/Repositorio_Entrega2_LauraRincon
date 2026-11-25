"""Rutas, carga de configuración y registro (log) compartidos por todas las etapas.

Al centralizar estos elementos, se evita que cada script maneje sus propias rutas o
registros de log. Así, todos trabajan con la misma configuración y guardan sus
registros en un único archivo `logs/pipeline.log`, facilitando el seguimiento de todo
el proceso.
"""

import hashlib
import json
import logging
import os
import platform
from pathlib import Path

# Raíz del repositorio: carpeta que contiene a src/
RAIZ = Path(__file__).resolve().parent.parent

RUTA_PARAMS = RAIZ / "config" / "params.json"
RUTA_CONTRATO = RAIZ / "contracts" / "data_contract.json"

DIR_RAW = RAIZ / "data" / "raw"
DIR_PROCESSED = RAIZ / "data" / "processed"
DIR_MODELOS = RAIZ / "models"
DIR_LOGS = RAIZ / "logs"
DIR_REPORTES = RAIZ / "reports"

RUTA_RAW = DIR_RAW / "iris.csv"
RUTA_TRAIN = DIR_PROCESSED / "train.csv"
RUTA_TEST = DIR_PROCESSED / "test.csv"
RUTA_MODELO = DIR_MODELOS / "model.pkl"
RUTA_BASELINE = DIR_MODELOS / "baseline.pkl"
RUTA_INFO_RUNS = DIR_MODELOS / "run_info.json"
RUTA_LOG = DIR_LOGS / "pipeline.log"
RUTA_REPORTE_VALIDACION = DIR_REPORTES / "validation_report.json"
RUTA_REPORTE_EVALUACION = DIR_REPORTES / "evaluation_report.json"


def cargar_json(ruta):
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def guardar_json(datos, ruta):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as archivo:
        json.dump(datos, archivo, ensure_ascii=False, indent=2)


def cargar_params():
    return cargar_json(RUTA_PARAMS)


def cargar_contrato():
    return cargar_json(RUTA_CONTRATO)


def sha256_archivo(ruta):
    """Huella digital del archivo: si los datos cambian, el hash cambia."""
    h = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(65536), b""):
            h.update(bloque)
    return h.hexdigest()


def versiones_entorno():
    """Versiones de Python y librerías que se registran como etiquetas en MLflow."""
    import mlflow
    import sklearn

    return {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "mlflow_version": mlflow.__version__,
        "sistema_operativo": platform.system(),
    }


def configurar_log(nombre):
    """Logger que escribe en consola y en logs/pipeline.log con fecha y etapa."""
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(nombre)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formato = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-9s | %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
    for manejador in (logging.StreamHandler(), logging.FileHandler(RUTA_LOG, encoding="utf-8")):
        manejador.setFormatter(formato)
        logger.addHandler(manejador)
    return logger


def preparar_entorno_mlflow():
    """Variables de entorno que silencian avisos informativos de MLflow y Git."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
