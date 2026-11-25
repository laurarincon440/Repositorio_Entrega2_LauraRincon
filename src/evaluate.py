"""Etapa 4 del pipeline: validación del modelo sobre el conjunto de prueba.

Esta etapa se encarga de evaluar la baseline y el modelo seleccionado utilizando las
mismas 30 flores de prueba y comparando sus resultados con los mínimos de calidad
definidos en config/params.json. Si el modelo seleccionado no cumple alguno de estos
requisitos, el proceso se detiene y genera un error.

También se verifica que el modelo cumpla con las reglas del contrato de datos. Para
ello, se comprueba que el ejemplo válido genere una predicción con el formato esperado
y que el ejemplo inválido sea rechazado antes de llegar al modelo.

Por último, las métricas obtenidas en las pruebas se agregan a los mismos registros de
MLflow que se crearon durante el entrenamiento, permitiendo consultar todo el proceso
en un solo lugar.
"""

import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src import utils
from src.predict import predecir

utils.preparar_entorno_mlflow()
import mlflow  # noqa: E402


def evaluar(modelo, X, y):
    pred = modelo.predict(X)
    return {
        "test_accuracy": float(accuracy_score(y, pred)),
        "test_f1_macro": float(f1_score(y, pred, average="macro")),
        "test_errores": int((pred != y).sum()),
        "matriz_confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def main():
    log = utils.configurar_log("evaluate")
    params = utils.cargar_params()
    contrato = utils.cargar_contrato()
    info = utils.cargar_json(utils.RUTA_INFO_RUNS)
    umbrales = params["umbrales_calidad"]

    test = pd.read_csv(utils.RUTA_TEST)
    X = test[contrato["modelo"]["orden_caracteristicas"]]
    y = test[contrato["variable_objetivo"]["nombre"]]

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    resultados = {}
    for clave, ruta in (("baseline", utils.RUTA_BASELINE), ("modelo_seleccionado", utils.RUTA_MODELO)):
        metricas = evaluar(joblib.load(ruta), X, y)
        metricas["cv_accuracy_mean"] = info[clave]["metricas_train"]["cv_accuracy_mean"]
        metricas["train_accuracy"] = info[clave]["metricas_train"]["train_accuracy"]
        metricas["brecha_train_test"] = metricas["train_accuracy"] - metricas["test_accuracy"]
        resultados[clave] = metricas
        with mlflow.start_run(run_id=info[clave]["run_id"]):
            mlflow.log_metrics({k: metricas[k] for k in
                                ("test_accuracy", "test_f1_macro", "test_errores", "brecha_train_test")})
            if clave == "baseline":
                mlflow.set_tag("validacion_modelo", "referencia")
        log.info("%-27s | test_acc=%.3f | f1_macro=%.3f | errores=%d/%d | brecha=%.3f",
                 info[clave]["run_name"], metricas["test_accuracy"], metricas["test_f1_macro"],
                 metricas["test_errores"], len(y), metricas["brecha_train_test"])

    seleccionado = resultados["modelo_seleccionado"]
    controles = {
        "accuracy_min": seleccionado["test_accuracy"] >= umbrales["accuracy_min"],
        "f1_macro_min": seleccionado["test_f1_macro"] >= umbrales["f1_macro_min"],
        "cv_accuracy_min": seleccionado["cv_accuracy_mean"] >= umbrales["cv_accuracy_min"],
        "no_inferior_al_baseline": (seleccionado["test_accuracy"]
                                    >= resultados["baseline"]["test_accuracy"]),
    }

    salida_valida = predecir(contrato["ejemplo_entrada_valida"])[0]
    salida_invalida = predecir(contrato["ejemplo_entrada_invalida"])[0]
    controles["contrato_ejemplo_valido_predicho"] = (
        salida_valida.get("prediction") in contrato["salida"]["campos"]["prediction"]["valores_posibles"])
    controles["contrato_ejemplo_invalido_rechazado"] = "errores" in salida_invalida
    log.info("Ejemplo válido del contrato   -> %s", salida_valida)
    log.info("Ejemplo inválido del contrato -> rechazado con %d errores",
             len(salida_invalida.get("errores", [])))

    for control, cumple in controles.items():
        (log.info if cumple else log.error)("Control %-36s %s", control, "OK" if cumple else "FALLA")
    aprobado = all(controles.values())

    utils.guardar_json({"umbrales": umbrales, "resultados": resultados, "controles": controles,
                        "ejemplo_valido_salida": salida_valida, "aprobado": aprobado},
                       utils.RUTA_REPORTE_EVALUACION)
    with mlflow.start_run(run_id=info["modelo_seleccionado"]["run_id"]):
        mlflow.log_artifact(str(utils.RUTA_REPORTE_EVALUACION), artifact_path="evaluacion")
        mlflow.set_tag("validacion_modelo", "aprobada" if aprobado else "rechazada")

    log.info("Validación del modelo: %s", "APROBADA" if aprobado else "RECHAZADA")
    sys.exit(0 if aprobado else 1)


if __name__ == "__main__":
    main()
