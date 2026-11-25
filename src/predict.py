"""Inferencia con el modelo entrenado aplicando el contrato de datos.

Cada registro se valida contra el contrato antes de llegar al modelo. Si incumple
alguna regla no se predice y se devuelven los errores. Así el modelo solo recibe
datos con las mismas columnas, tipos y rangos con los que fue entrenado, lo que
evita el sesgo entre entrenamiento y servicio (training-serving skew).

Uso:
  python -m src.predict --ejemplo valida
  python -m src.predict --ejemplo invalida
  python -m src.predict --archivo solicitud.json
"""

import argparse
import json

import joblib
import pandas as pd

from src import utils
from src.validate_data import validar_solicitud


def predecir(datos):
    """Recibe un objeto o una lista de objetos JSON y devuelve una salida por registro."""
    contrato = utils.cargar_contrato()
    columnas = contrato["modelo"]["orden_caracteristicas"]
    etiquetas = contrato["variable_objetivo"]["etiquetas"]
    registros = datos if isinstance(datos, list) else [datos]
    modelo = joblib.load(utils.RUTA_MODELO)

    salidas = []
    for registro, validacion in zip(registros, validar_solicitud(registros, contrato)):
        if not validacion["aprobado"]:
            salidas.append({"errores": validacion["errores"]})
            continue
        fila = pd.DataFrame([registro], columns=columnas).astype(float)
        clase = int(modelo.predict(fila)[0])
        salidas.append({"prediction": etiquetas[str(clase)], "class_id": clase})
    return salidas


def main():
    parser = argparse.ArgumentParser(description="Predicción de especie Iris según el contrato")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--ejemplo", choices=["valida", "invalida"],
                       help="Usa el ejemplo de entrada válida o inválida del contrato")
    grupo.add_argument("--archivo", help="Archivo JSON con uno o varios registros")
    args = parser.parse_args()

    if args.ejemplo:
        datos = utils.cargar_contrato()[f"ejemplo_entrada_{args.ejemplo}"]
    else:
        datos = utils.cargar_json(args.archivo)
    print(json.dumps(predecir(datos), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
