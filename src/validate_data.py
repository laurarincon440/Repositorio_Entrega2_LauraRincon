"""Etapa 2 del pipeline: validación de datos de entrada sin librerías externas.

Solo usa la librería estándar de Python (csv, json, math, argparse). Las reglas
no están escritas en el código, se leen de contracts/data_contract.json. De esta forma,
tanto el entrenamiento como la inferencia utilizan los mismos criterios para validar
los datos.

Se verifican en cada registro las reglas definidas en el contrato (columnas, nulo,
tipo, rango, dominio y relación).
A nivel de archivo se revisa que tenga registros y se reportan como advertencia
(no bloquean) los registros duplicados.
Además se tienen en cuenta las reglas de conjunto, que comparan los archivos entre sí.

Uso:
  python -m src.validate_data                         valida data/raw, train y test
  python -m src.validate_data --entrada archivo.json  valida una solicitud de inferencia
  python -m src.validate_data --ejemplos              valida los ejemplos del contrato
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime

from src import utils

# Textos que se interpretan como valor faltante al leer un CSV o un JSON
TEXTOS_NULOS = {"", "na", "n/a", "nan", "null", "none", "nulo", "-"}


def es_nulo(valor):
    if valor is None:
        return True
    if isinstance(valor, float) and math.isnan(valor):
        return True
    if isinstance(valor, str) and valor.strip().lower() in TEXTOS_NULOS:
        return True
    return False


def convertir_numero(valor, origen):
    """Devuelve el valor como float o None si no es un número válido.
    """

    if isinstance(valor, bool):
        return None
    if origen == "json":
        if isinstance(valor, (int, float)):
            numero = float(valor)
        else:
            return None
    else:
        try:
            numero = float(str(valor).strip())
        except ValueError:
            return None
    return numero if math.isfinite(numero) else None


def convertir_entero(valor, origen):
    numero = convertir_numero(valor, origen)
    if numero is None or not numero.is_integer():
        return None
    return int(numero)


def error(campo, regla, mensaje, valor):
    return {"campo": campo, "regla": regla, "mensaje": mensaje, "valor": repr(valor)}


def validar_registro(registro, contrato, incluir_objetivo=False, origen="json"):
    """Valida un registro (diccionario columna -> valor) y devuelve la lista de errores.

    incluir_objetivo=True se usa para los archivos de entrenamiento y prueba, que
    traen la columna species. En inferencia (False) esa columna está prohibida.
    """
    campos = contrato["entrada"]["campos"]
    objetivo = contrato["variable_objetivo"]
    esperadas = list(campos) + ([objetivo["nombre"]] if incluir_objetivo else [])
    errores = []

    for columna in esperadas:
        if columna not in registro:
            errores.append(error(columna, "columnas", "Falta la columna requerida", None))
    if not contrato["entrada"]["campos_adicionales_permitidos"]:
        for columna in registro:
            if columna not in esperadas:
                motivo = ("Campo no permitido en inferencia: es la variable objetivo (fuga de datos)"
                          if columna == objetivo["nombre"] else "Campo no definido en el contrato")
                errores.append(error(columna, "columnas", motivo, registro[columna]))

    numeros = {}
    for nombre, regla in campos.items():
        if nombre not in registro:
            continue
        valor = registro[nombre]
        if es_nulo(valor):
            if not regla["admite_nulos"]:
                errores.append(error(nombre, "nulo", "Valor nulo o vacío", valor))
            continue
        numero = convertir_numero(valor, origen)
        if numero is None:
            errores.append(error(nombre, "tipo", f"Se esperaba {regla['tipo']} y llegó "
                                 f"{type(valor).__name__}", valor))
            continue
        if not regla["minimo"] <= numero <= regla["maximo"]:
            errores.append(error(nombre, "rango", f"Fuera del rango lógico "
                                 f"[{regla['minimo']}, {regla['maximo']}] {regla['unidad']}", valor))
            continue
        numeros[nombre] = numero

    # Las reglas entre campos solo se evalúan si ambos valores ya pasaron las
    # validaciones individuales, para no reportar el mismo problema dos veces

    for relacion in contrato["entrada"]["reglas_entre_campos"]:
        menor, mayor = relacion["campo_menor"], relacion["campo_mayor"]
        if menor in numeros and mayor in numeros and not numeros[menor] < numeros[mayor]:
            errores.append(error(f"{menor}/{mayor}", "relacion", relacion["descripcion"],
                                 (numeros[menor], numeros[mayor])))

    if incluir_objetivo and objetivo["nombre"] in registro:
        valor = registro[objetivo["nombre"]]
        if es_nulo(valor):
            errores.append(error(objetivo["nombre"], "nulo", "Valor nulo o vacío", valor))
        else:
            clase = convertir_entero(valor, origen)
            if clase is None:
                errores.append(error(objetivo["nombre"], "tipo", "Se esperaba un entero", valor))
            elif clase not in objetivo["valores_permitidos"]:
                errores.append(error(objetivo["nombre"], "dominio",
                                     f"Clase fuera de {objetivo['valores_permitidos']}", valor))
    return errores


def validar_csv(ruta, contrato):
    """Valida un archivo CSV de entrenamiento o prueba.

    Devuelve el reporte del archivo y sus filas, que luego usan las reglas de conjunto.
    """
    reporte = {"archivo": str(ruta.relative_to(utils.RAIZ)).replace("\\", "/"),
               "sha256": utils.sha256_archivo(ruta), "registros": 0,
               "errores": [], "advertencias": []}
    with open(ruta, newline="", encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    vistos = {}
    # La fila 1 del archivo es el encabezado, por eso los datos empiezan en la 2
    for numero_fila, fila in enumerate(filas, start=2):
        reporte["registros"] += 1
        for e in validar_registro(fila, contrato, incluir_objetivo=True, origen="csv"):
            reporte["errores"].append({"fila": numero_fila, **e})
        clave = tuple(fila.values())
        if clave in vistos:
            reporte["advertencias"].append(
                f"Fila {numero_fila} duplica la fila {vistos[clave]} {list(fila.values())}")
        else:
            vistos[clave] = numero_fila
    if reporte["registros"] == 0:
        reporte["errores"].append({"fila": None, "campo": None, "regla": "vacio",
                                   "mensaje": "El archivo no contiene registros", "valor": None})
    reporte["aprobado"] = not reporte["errores"]
    return reporte, filas


def clave_flor(fila, contrato):
    """Representa una flor como tupla de números para compararla entre archivos.

    Devuelve None si algún valor no es numérico, esos casos ya los reporta la
    validación por registro.
    """
    columnas = list(contrato["entrada"]["campos"]) + [contrato["variable_objetivo"]["nombre"]]
    valores = [convertir_numero(fila.get(c), "csv") for c in columnas]
    return None if None in valores else tuple(valores)


def validar_conjuntos(raw, train, test, contrato):
    """Reglas que comparan los archivos entre sí y que una fila aislada no puede
    revelar."""

    reglas = {r["id"] for r in contrato["reglas_de_conjunto"]}
    objetivo = contrato["variable_objetivo"]
    errores = []

    if "clases_completas" in reglas:
        for nombre, filas in (("train.csv", train), ("test.csv", test)):
            presentes = {convertir_entero(f.get(objetivo["nombre"]), "csv") for f in filas}
            faltantes = [c for c in objetivo["valores_permitidos"] if c not in presentes]
            if faltantes:
                etiquetas = [objetivo["etiquetas"][str(c)] for c in faltantes]
                errores.append({"regla": "clases_completas", "archivo": nombre,
                                "mensaje": f"Faltan especies: {etiquetas}"})

    if "sin_solapamiento_train_test" in reglas:
        # Una flor que ya viene repetida en el dataset original puede quedar legítimamente
        # en ambos conjuntos. Cualquier otra coincidencia indica que se copiaron
        # registros

        conteo_raw = {}
        for f in raw:
            clave = clave_flor(f, contrato)
            conteo_raw[clave] = conteo_raw.get(clave, 0) + 1
        claves_train = {clave_flor(f, contrato) for f in train} - {None}
        compartidas = [clave_flor(f, contrato) for f in test
                       if clave_flor(f, contrato) in claves_train]
        no_explicadas = [c for c in compartidas if conteo_raw.get(c, 0) < 2]
        if no_explicadas:
            errores.append({"regla": "sin_solapamiento_train_test", "archivo": "train.csv/test.csv",
                            "mensaje": f"{len(no_explicadas)} de {len(test)} flores de prueba también "
                                       "están en entrenamiento (fuga de datos)"})
    return errores


def validar_solicitud(datos, contrato):
    """Valida una solicitud de inferencia: un objeto JSON o una lista de objetos."""
    registros = datos if isinstance(datos, list) else [datos]
    resultados = []
    for indice, registro in enumerate(registros):
        if not isinstance(registro, dict):
            errores = [error(None, "formato", "Cada registro debe ser un objeto JSON", registro)]
        else:
            errores = validar_registro(registro, contrato, incluir_objetivo=False, origen="json")
        resultados.append({"registro": indice, "aprobado": not errores, "errores": errores})
    return resultados


def resumir_errores(errores):
    """Cuenta los errores por regla, útil para leer el reporte de un vistazo."""
    conteo = {}
    for e in errores:
        conteo[e["regla"]] = conteo.get(e["regla"], 0) + 1
    return conteo


def validar_archivos_pipeline(log):
    contrato = utils.cargar_contrato()
    resultados = [validar_csv(ruta, contrato)
                  for ruta in (utils.RUTA_RAW, utils.RUTA_TRAIN, utils.RUTA_TEST)]
    reportes = [reporte for reporte, _ in resultados]
    for r in reportes:
        r["errores_por_regla"] = resumir_errores(r["errores"])
        estado = "APROBADO" if r["aprobado"] else "RECHAZADO"
        log.info("%s | %s | %d registros | %d errores %s | %d advertencias", estado,
                 r["archivo"], r["registros"], len(r["errores"]), r["errores_por_regla"] or "",
                 len(r["advertencias"]))
        for e in r["errores"][:10]:
            log.error("  fila %s | %s | %s | %s | valor=%s", e["fila"], e["campo"], e["regla"],
                      e["mensaje"], e["valor"])
        for a in r["advertencias"]:
            log.warning("  %s", a)

    errores_conjunto = validar_conjuntos(*(filas for _, filas in resultados), contrato)
    log.info("%s | reglas de conjunto: %s", "RECHAZADO" if errores_conjunto else "APROBADO",
             ", ".join(r["id"] for r in contrato["reglas_de_conjunto"]))
    for e in errores_conjunto:
        log.error("  %s | %s | %s", e["archivo"], e["regla"], e["mensaje"])

    aprobado = all(r["aprobado"] for r in reportes) and not errores_conjunto
    utils.guardar_json({
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "contrato": f"{contrato['nombre']} v{contrato['version']}",
        "reglas": ["columnas", "nulo", "tipo", "rango", "dominio", "relacion"]
                  + [r["id"] for r in contrato["reglas_de_conjunto"]],
        "aprobado": aprobado,
        "archivos": reportes,
        "errores_de_conjunto": errores_conjunto,
    }, utils.RUTA_REPORTE_VALIDACION)
    log.info("Reporte guardado en %s", utils.RUTA_REPORTE_VALIDACION.relative_to(utils.RAIZ))
    return aprobado


def validar_ejemplos_contrato(log):
    """Comprueba que el contrato sea coherente. El ejemplo válido debe pasar y el
    inválido debe fallar exactamente con los errores que el contrato anuncia."""

    contrato = utils.cargar_contrato()
    valido = validar_solicitud(contrato["ejemplo_entrada_valida"], contrato)[0]
    invalido = validar_solicitud(contrato["ejemplo_entrada_invalida"], contrato)[0]
    obtenidos = {(e["campo"], e["regla"]) for e in invalido["errores"]}
    esperados = {(e["campo"], e["regla"]) for e in contrato["errores_esperados_entrada_invalida"]}

    log.info("Ejemplo válido   -> %s", "aprobado" if valido["aprobado"] else valido["errores"])
    log.info("Ejemplo inválido -> %d errores detectados", len(invalido["errores"]))
    for e in invalido["errores"]:
        log.info("  %s | %s | %s | valor=%s", e["campo"], e["regla"], e["mensaje"], e["valor"])
    coherente = valido["aprobado"] and not invalido["aprobado"] and obtenidos == esperados
    log.info("Contrato coherente con el validador: %s", "sí" if coherente else "no")
    return coherente


def main():
    parser = argparse.ArgumentParser(description="Validación de datos contra el contrato")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--entrada", help="Archivo JSON con una solicitud de inferencia")
    grupo.add_argument("--ejemplos", action="store_true",
                       help="Valida los ejemplos válido e inválido del contrato")
    args = parser.parse_args()
    log = utils.configurar_log("validate")

    if args.entrada:
        with open(args.entrada, encoding="utf-8") as archivo:
            resultados = validar_solicitud(json.load(archivo), utils.cargar_contrato())
        print(json.dumps(resultados, ensure_ascii=False, indent=2))
        correcto = all(r["aprobado"] for r in resultados)
    elif args.ejemplos:
        correcto = validar_ejemplos_contrato(log)
    else:
        correcto = validar_archivos_pipeline(log) and validar_ejemplos_contrato(log)

    # Código de salida 1 detiene el pipeline (Makefile y run_pipeline.py)
    sys.exit(0 if correcto else 1)

if __name__ == "__main__":
    main()
