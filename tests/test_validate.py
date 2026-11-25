"""Pruebas unitarias del validador de datos (src/validate_data.py).

Cada prueba construye un registro o un grupo de archivos con un único problema y
verifica que el validador lo detecte con la regla correcta. No dependen de que el
pipeline se haya ejecutado antes.
"""

import pytest

from src import utils
from src.validate_data import es_nulo, validar_conjuntos, validar_registro, validar_solicitud

CONTRATO = utils.cargar_contrato()
VALIDO = CONTRATO["ejemplo_entrada_valida"]


def reglas(errores):
    return {(e["campo"], e["regla"]) for e in errores}


def test_ejemplo_valido_del_contrato_pasa():
    assert validar_registro(VALIDO, CONTRATO) == []


def test_ejemplo_invalido_del_contrato_falla_con_los_errores_esperados():
    errores = validar_registro(CONTRATO["ejemplo_entrada_invalida"], CONTRATO)
    esperados = {(e["campo"], e["regla"]) for e in CONTRATO["errores_esperados_entrada_invalida"]}
    assert reglas(errores) == esperados


@pytest.mark.parametrize("valor", [None, "", "  ", "NA", "nan", "null", float("nan")])
def test_detecta_nulos(valor):
    assert es_nulo(valor)
    assert ("sepal_width", "nulo") in reglas(validar_registro({**VALIDO, "sepal_width": valor}, CONTRATO))


@pytest.mark.parametrize("valor", ["5.1", "cinco", True, [5.1], {"cm": 5.1}])
def test_rechaza_tipos_no_numericos_en_json(valor):
    errores = validar_registro({**VALIDO, "sepal_length": valor}, CONTRATO)
    assert ("sepal_length", "tipo") in reglas(errores)


def test_acepta_texto_numerico_en_csv():
    fila = {k: str(v) for k, v in VALIDO.items()}
    assert validar_registro(fila, CONTRATO, origen="csv") == []


@pytest.mark.parametrize("campo,valor", [("sepal_length", 3.9), ("sepal_length", 8.1),
                                         ("petal_width", 0.0), ("petal_length", -1.0),
                                         ("sepal_width", float("inf"))])
def test_detecta_valores_fuera_de_rango_o_infinitos(campo, valor):
    errores = validar_registro({**VALIDO, campo: valor}, CONTRATO)
    assert reglas(errores) & {(campo, "rango"), (campo, "tipo")}


def test_detecta_columna_faltante_y_columna_extra():
    registro = {k: v for k, v in VALIDO.items() if k != "petal_width"}
    registro["color"] = "morado"
    assert reglas(validar_registro(registro, CONTRATO)) == {("petal_width", "columnas"),
                                                            ("color", "columnas")}


def test_rechaza_variable_objetivo_en_inferencia():
    errores = validar_registro({**VALIDO, "species": 1}, CONTRATO)
    assert ("species", "columnas") in reglas(errores)


@pytest.mark.parametrize("valor,regla", [("3", "dominio"), ("1.5", "tipo"), ("", "nulo")])
def test_valida_variable_objetivo_en_entrenamiento(valor, regla):
    fila = {**{k: str(v) for k, v in VALIDO.items()}, "species": valor}
    errores = validar_registro(fila, CONTRATO, incluir_objetivo=True, origen="csv")
    assert ("species", regla) in reglas(errores)


def test_detecta_columnas_intercambiadas_con_la_regla_entre_campos():
    registro = {**VALIDO, "sepal_length": 4.5, "petal_length": 6.0}
    assert ("petal_length/sepal_length", "relacion") in reglas(validar_registro(registro, CONTRATO))


def test_solicitud_por_lotes_valida_cada_registro():
    resultados = validar_solicitud([VALIDO, CONTRATO["ejemplo_entrada_invalida"], "texto"], CONTRATO)
    assert [r["aprobado"] for r in resultados] == [True, False, False]


def flor(sl, sw, pl, pw, especie):
    return {"sepal_length": str(sl), "sepal_width": str(sw), "petal_length": str(pl),
            "petal_width": str(pw), "species": str(especie)}


SETOSA, VERSICOLOR, VIRGINICA = flor(5.1, 3.5, 1.4, 0.2, 0), flor(6.0, 2.9, 4.5, 1.4, 1), flor(6.5, 3.0, 5.5, 2.0, 2)
OTRA_SETOSA, OTRA_VERSICOLOR, OTRA_VIRGINICA = (flor(4.9, 3.0, 1.4, 0.2, 0), flor(5.7, 2.8, 4.1, 1.3, 1),
                                                flor(6.3, 3.3, 6.0, 2.5, 2))


def test_conjuntos_correctos_no_generan_errores():
    train, test = [SETOSA, VERSICOLOR, VIRGINICA], [OTRA_SETOSA, OTRA_VERSICOLOR, OTRA_VIRGINICA]
    assert validar_conjuntos(train + test, train, test, CONTRATO) == []


def test_detecta_especie_ausente_en_entrenamiento():
    train, test = [SETOSA, VERSICOLOR], [OTRA_SETOSA, OTRA_VERSICOLOR, OTRA_VIRGINICA]
    errores = validar_conjuntos(train + test + [VIRGINICA], train, test, CONTRATO)
    assert [e["regla"] for e in errores] == ["clases_completas"]


def test_detecta_flores_de_prueba_copiadas_en_entrenamiento():
    test = [OTRA_SETOSA, OTRA_VERSICOLOR, OTRA_VIRGINICA]
    train = [SETOSA, VERSICOLOR, VIRGINICA] + test
    errores = validar_conjuntos(train, train, test, CONTRATO)
    assert [e["regla"] for e in errores] == ["sin_solapamiento_train_test"]


def test_tolera_una_flor_duplicada_en_el_dataset_original():
    train, test = [SETOSA, VERSICOLOR, VIRGINICA], [SETOSA, OTRA_VERSICOLOR, OTRA_VIRGINICA]
    raw = [SETOSA, SETOSA, VERSICOLOR, VIRGINICA, OTRA_VERSICOLOR, OTRA_VIRGINICA]
    assert validar_conjuntos(raw, train, test, CONTRATO) == []
