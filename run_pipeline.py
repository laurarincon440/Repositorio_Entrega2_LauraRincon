"""Orquestador del pipeline: ejecuta todas las etapas con un solo comando.

    python run_pipeline.py              ejecuta el pipeline completo
    python run_pipeline.py --reiniciar  borra los resultados previos y reproduce desde cero
    python run_pipeline.py --sin-preparar  omite prepare y valida los datos tal como están
                                           (se usa con romper_datos.py)

Etapas, en orden:
  1. prepare   extracción y partición 80/20 de los datos
  2. validate  validación de datos contra el contrato (tipos, nulos, rangos)
  3. train     entrenamiento de la baseline y del modelo seleccionado con registro en MLflow
  4. evaluate  validación del modelo con umbrales de calidad sobre el conjunto de prueba
  5. test      pruebas automatizadas con pytest

Cada etapa corre en un proceso independiente. Si una termina con error, el
pipeline se detiene y las etapas siguientes no se ejecutan, de modo que nunca se
entrena con datos rechazados ni se da por válido un modelo que no pasó los umbrales.

Es el equivalente en Python de `make pipeline`, útil en Windows, donde make no
viene instalado por defecto.
"""

import argparse
import shutil
import subprocess
import sys
import time

from src import utils

ETAPAS = [
    ("prepare", [sys.executable, "-m", "src.prepare"]),
    ("validate", [sys.executable, "-m", "src.validate_data"]),
    ("train", [sys.executable, "-m", "src.train"]),
    ("evaluate", [sys.executable, "-m", "src.evaluate"]),
    ("test", [sys.executable, "-m", "pytest", "-q", "tests"]),
]

# Resultados generados por el pipeline; se eliminan con --reiniciar
GENERADOS = ["mlflow.db", "mlruns", "data", "models", "reports", "logs"]


def reiniciar():
    for nombre in GENERADOS:
        ruta = utils.RAIZ / nombre
        if ruta.is_dir():
            shutil.rmtree(ruta)
        elif ruta.exists():
            ruta.unlink()


def main():
    parser = argparse.ArgumentParser(description="Pipeline único de clasificación de Iris")
    parser.add_argument("--reiniciar", action="store_true",
                        help="Elimina mlflow.db, mlruns, data, models, reports y logs antes de ejecutar")
    parser.add_argument("--sin-preparar", action="store_true",
                        help="Omite la etapa prepare para validar los datos actuales sin regenerarlos")
    args = parser.parse_args()
    etapas = [e for e in ETAPAS if not (args.sin_preparar and e[0] == "prepare")]

    if args.reiniciar:
        reiniciar()
    log = utils.configurar_log("pipeline")
    if args.reiniciar:
        log.info("Resultados previos eliminados: %s", ", ".join(GENERADOS))

    log.info("Inicio del pipeline (%d etapas)", len(etapas))
    inicio_total = time.time()
    for numero, (nombre, comando) in enumerate(etapas, start=1):
        log.info("Etapa %d/%d: %s", numero, len(etapas), nombre)
        inicio = time.time()
        resultado = subprocess.run(comando, cwd=utils.RAIZ)
        duracion = time.time() - inicio
        if resultado.returncode != 0:
            log.error("Etapa %s falló (código %d) tras %.1f s. Pipeline detenido.",
                      nombre, resultado.returncode, duracion)
            sys.exit(resultado.returncode)
        log.info("Etapa %s completada en %.1f s", nombre, duracion)

    log.info("Pipeline completado en %.1f s. Consulte los runs con: "
             "mlflow ui --backend-store-uri sqlite:///mlflow.db", time.time() - inicio_total)


if __name__ == "__main__":
    main()
