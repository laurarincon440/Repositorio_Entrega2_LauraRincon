"""Pruebas sobre los resultados del pipeline (datos procesados y modelo).

Se ejecutan como última etapa del pipeline, después de prepare, validate, train
y evaluate, dado que revisan los archivos que esas etapas generan.
"""

import joblib
import pandas as pd

from src import utils
from src.predict import predecir

CONTRATO = utils.cargar_contrato()
COLUMNAS = CONTRATO["modelo"]["orden_caracteristicas"]


def test_particion_reproducible():
    train, test = pd.read_csv(utils.RUTA_TRAIN), pd.read_csv(utils.RUTA_TEST)
    assert (len(train), len(test)) == (120, 30)
    # La partición estratificada conserva la proporción de especies: 40/40/40 y 10/10/10
    assert train["species"].value_counts().sort_index().tolist() == [40, 40, 40]
    assert test["species"].value_counts().sort_index().tolist() == [10, 10, 10]


def test_columnas_coinciden_con_el_contrato():
    assert list(pd.read_csv(utils.RUTA_TRAIN).columns) == COLUMNAS + ["species"]


def test_validacion_de_datos_aprobada():
    assert utils.cargar_json(utils.RUTA_REPORTE_VALIDACION)["aprobado"]


def test_modelo_con_hiperparametros_del_modelo_seleccionado():
    modelo = joblib.load(utils.RUTA_MODELO)
    config = utils.cargar_params()["modelo_seleccionado"]
    assert (modelo.n_estimators, modelo.max_depth) == (config["n_estimators"], config["max_depth"])
    assert list(modelo.feature_names_in_) == COLUMNAS


def test_accuracy_supera_el_umbral():
    modelo = joblib.load(utils.RUTA_MODELO)
    test = pd.read_csv(utils.RUTA_TEST)
    umbral = utils.cargar_params()["umbrales_calidad"]["accuracy_min"]
    assert modelo.score(test[COLUMNAS], test["species"]) >= umbral


def test_salida_de_inferencia_cumple_el_contrato():
    salida = predecir(CONTRATO["ejemplo_entrada_valida"])[0]
    assert set(salida) == set(CONTRATO["salida"]["campos"])
    assert salida["prediction"] == CONTRATO["variable_objetivo"]["etiquetas"][str(salida["class_id"])]


def test_inferencia_rechaza_la_entrada_invalida():
    assert "errores" in predecir(CONTRATO["ejemplo_entrada_invalida"])[0]
