"""Etapa 3 del pipeline: entrenamiento y registro en MLflow.

Entrena dos modelos con los mismos datos y semillas:
  - Baseline (100 árboles, sin límite de profundidad): punto de referencia.
  - Modelo seleccionado en la Actividad 1.1 (120 árboles, profundidad 4).

Solo usa data/processed/train.csv. El conjunto de prueba se reserva para la
etapa de evaluación por lo que ninguna decisión de entrenamiento lo ve.

Antes de entrenar comprueba que el último reporte de validación esté aprobado y
que train.csv no haya cambiado desde que se validó (mismo SHA-256).
"""

import sys
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

from src import utils

utils.preparar_entorno_mlflow()
import mlflow  # noqa: E402  (se importa después de configurar las variables de entorno)
import mlflow.sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402


def verificar_validacion(log):
    """Bloquea el entrenamiento si los datos no pasaron la validación."""
    if not utils.RUTA_REPORTE_VALIDACION.exists():
        log.error("No existe %s. Ejecute primero la etapa validate.", utils.RUTA_REPORTE_VALIDACION.name)
        return False
    reporte = utils.cargar_json(utils.RUTA_REPORTE_VALIDACION)
    if not reporte["aprobado"]:
        log.error("La validación de datos fue rechazada. No se entrena el modelo.")
        return False
    hash_validado = {a["archivo"]: a["sha256"] for a in reporte["archivos"]}["data/processed/train.csv"]
    if hash_validado != utils.sha256_archivo(utils.RUTA_TRAIN):
        log.error("train.csv cambió después de la validación. Vuelva a ejecutar validate.")
        return False
    return True


def entrenar_y_registrar(config, X, y, params, contrato, etiquetas, log):
    semilla = params["entrenamiento"]["model_seed"]
    folds = params["entrenamiento"]["cv_folds"]

    with mlflow.start_run(run_name=config["run_name"]) as run:
        mlflow.log_params({
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "model_seed": semilla,
            "split_seed": params["datos"]["split_seed"],
            "test_size": params["datos"]["test_size"],
            "estratificado": params["datos"]["estratificar"],
            "cv_folds": folds,
            "train_registros": len(X),
            "train_sha256": utils.sha256_archivo(utils.RUTA_TRAIN),
            "contrato_version": contrato["version"],
        })
        mlflow.set_tags(etiquetas)

        def nuevo_modelo():
            return RandomForestClassifier(n_estimators=config["n_estimators"],
                                          max_depth=config["max_depth"], random_state=semilla)

        # La validación cruzada se hace solo sobre train para no usar el conjunto de prueba
        cv = cross_val_score(nuevo_modelo(), X, y, cv=folds)

        modelo = nuevo_modelo()
        inicio = time.time()
        modelo.fit(X, y)
        tiempo = time.time() - inicio

        pred_train = modelo.predict(X)
        metricas = {
            "train_accuracy": float((pred_train == y).mean()),
            "train_f1_macro": float(f1_score(y, pred_train, average="macro")),
            "cv_accuracy_mean": float(cv.mean()),
            "cv_accuracy_std": float(cv.std()),
            "train_time_s": tiempo,
        }
        mlflow.log_metrics(metricas)

        # La firma (signature) documenta en MLflow las mismas columnas y tipos del contrato
        firma = infer_signature(X, modelo.predict(X))
        mlflow.sklearn.log_model(modelo, name="iris_model", signature=firma,
                                 input_example=X.head(3),
                                 skops_trusted_types=["sklearn.tree._tree.Tree"])
        mlflow.log_artifact(str(utils.RUTA_CONTRATO), artifact_path="contrato")
        mlflow.log_artifact(str(utils.RUTA_REPORTE_VALIDACION), artifact_path="validacion")
        mlflow.log_artifact(str(utils.RUTA_PARAMS), artifact_path="config")
        mlflow.log_artifact(str(utils.RAIZ / "requirements.txt"))

        log.info("%-27s | train_acc=%.3f | cv=%.3f ± %.3f | run_id=%s", config["run_name"],
                 metricas["train_accuracy"], metricas["cv_accuracy_mean"],
                 metricas["cv_accuracy_std"], run.info.run_id)
        return modelo, run.info.run_id, metricas


def main():
    log = utils.configurar_log("train")
    if not verificar_validacion(log):
        sys.exit(1)

    params = utils.cargar_params()
    contrato = utils.cargar_contrato()
    columnas = contrato["modelo"]["orden_caracteristicas"]
    objetivo = contrato["variable_objetivo"]["nombre"]

    train = pd.read_csv(utils.RUTA_TRAIN)
    X, y = train[columnas], train[objetivo]

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experimento"])
    etiquetas = {
        "autor": "Laura Tatiana Rincón López",
        "dataset": "Iris (sklearn.datasets.load_iris, 150 registros)",
        "modelo": "RandomForestClassifier",
        "etapa": "Seccion_2_pipeline",
        **utils.versiones_entorno(),
    }

    utils.DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    info = {}
    for clave, ruta in (("baseline", utils.RUTA_BASELINE), ("modelo_seleccionado", utils.RUTA_MODELO)):
        modelo, run_id, metricas = entrenar_y_registrar(params[clave], X, y, params, contrato,
                                                        {**etiquetas, "rol": clave}, log)
        joblib.dump(modelo, ruta)
        info[clave] = {"run_name": params[clave]["run_name"], "run_id": run_id,
                       "artefacto": str(ruta.relative_to(utils.RAIZ)), "metricas_train": metricas}

    utils.guardar_json(info, utils.RUTA_INFO_RUNS)
    log.info("Modelos guardados en %s y run IDs en %s", utils.DIR_MODELOS.name,
             utils.RUTA_INFO_RUNS.name)


if __name__ == "__main__":
    main()
