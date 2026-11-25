"""Etapa 1 del pipeline: extracción y partición de los datos.

Carga el dataset Iris incluido en scikit-learn, renombra las columnas a los
nombres que define el contrato de datos y guarda:
  - data/raw/iris.csv              dataset completo (150 registros)
  - data/processed/train.csv       80 % para entrenamiento
  - data/processed/test.csv        20 % para prueba

La partición usa la misma semilla (42) y proporción (80/20) de la Actividad 1.1,
pero ahora es estratificada pues conserva en train y test la proporción de cada
especie (40/40/40 y 10/10/10), como se propuso en clase. En la Actividad
1.1 la partición no era estratificada y la prueba quedó con 10, 9 y 11 flores.
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src import utils

# Nombres originales de scikit-learn y nombres del contrato (sin espacios ni
# unidades, compatibles con la API de la Semana 3)

NOMBRES_COLUMNAS = {
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
}


def main():
    log = utils.configurar_log("prepare")
    params = utils.cargar_params()["datos"]

    iris = load_iris(as_frame=True)
    datos = iris.data.rename(columns=NOMBRES_COLUMNAS)
    datos["species"] = iris.target

    utils.DIR_RAW.mkdir(parents=True, exist_ok=True)
    utils.DIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    datos.to_csv(utils.RUTA_RAW, index=False)
    log.info("Dataset crudo guardado: %s (%d registros, %d columnas)",
             utils.RUTA_RAW.relative_to(utils.RAIZ), *datos.shape)

    train, test = train_test_split(
        datos, test_size=params["test_size"], random_state=params["split_seed"],
        stratify=datos["species"] if params["estratificar"] else None,
    )
    train.to_csv(utils.RUTA_TRAIN, index=False)
    test.to_csv(utils.RUTA_TEST, index=False)

    conteo_test = test["species"].value_counts().sort_index().to_dict()
    log.info("Partición %d/%d con split_seed=%d (estratificada: %s) -> train: %d registros | "
             "test: %d registros", round((1 - params["test_size"]) * 100),
             round(params["test_size"] * 100), params["split_seed"],
             "sí" if params["estratificar"] else "no", len(train), len(test))
    log.info("Clases en test (0=setosa, 1=versicolor, 2=virginica): %s", conteo_test)
    log.info("SHA-256 train.csv: %s", utils.sha256_archivo(utils.RUTA_TRAIN))


if __name__ == "__main__":
    main()
