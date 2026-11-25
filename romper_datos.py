"""Sabotajes del juego "¡Rompe el pipeline!" visto en clase.

Regenera datos limpios con la etapa prepare y luego daña data/processed/train.csv
según el escenario elegido. Después se ejecuta el pipeline sin la etapa prepare
para ver qué control lo detiene:

    python romper_datos.py 3
    python run_pipeline.py --sin-preparar

Para volver a los datos correctos basta con ejecutar el pipeline completo
(python run_pipeline.py), porque prepare regenera los archivos.

Escenarios y control que los detiene en este repositorio:
  1  "cinco" en lugar de un número               validate_data · tipo
  2  celda vacía en petal_width                  validate_data · nulo
  3  un registro digitado en milímetros          validate_data · rango (4 errores)
  4  especie con código 3                        validate_data · dominio
  5  la extracción no trajo flores virginica     validate_data · clases_completas
  6  40 % de las etiquetas mal asignadas         evaluate · umbrales de accuracy
  7  las 30 flores de prueba copiadas en train   validate_data · sin_solapamiento_train_test
"""

import argparse

import numpy as np
import pandas as pd

from src import prepare, utils

ESCENARIOS = {
    1: "«cinco» en lugar de un número",
    2: "Celda vacía en petal_width",
    3: "Un registro digitado en milímetros",
    4: "Especie con código 3",
    5: "La extracción no trajo flores virginica",
    6: "40 % de las etiquetas mal asignadas",
    7: "Las 30 flores de prueba copiadas en entrenamiento",
}


def sabotear(escenario, train, test):
    medidas = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    train = train.astype(object)
    if escenario == 1:
        train.loc[0, "sepal_length"] = "cinco"
    elif escenario == 2:
        train.loc[0, "petal_width"] = ""
    elif escenario == 3:
        # Primera flor con pétalo de al menos 1 cm, para que las cuatro medidas en mm superen el máximo
        fila = train.index[train["petal_width"].astype(float) >= 1.0][0]
        train.loc[fila, medidas] = train.loc[fila, medidas].astype(float) * 10
    elif escenario == 4:
        train.loc[0, "species"] = 3
    elif escenario == 5:
        train = train[train["species"] != 2]
    elif escenario == 6:
        # Se cambia la etiqueta del 40 % de las flores por otra especie distinta
        rng = np.random.default_rng(0)
        indices = rng.choice(len(train), size=int(0.4 * len(train)), replace=False)
        for i in indices:
            actual = int(train.iloc[i]["species"])
            train.iloc[i, train.columns.get_loc("species")] = rng.choice(
                [c for c in (0, 1, 2) if c != actual])
    elif escenario == 7:
        train = pd.concat([train, test], ignore_index=True)
    return train


def main():
    parser = argparse.ArgumentParser(description="Daña train.csv según un escenario del juego")
    parser.add_argument("escenario", type=int, choices=sorted(ESCENARIOS))
    args = parser.parse_args()

    prepare.main()
    train = pd.read_csv(utils.RUTA_TRAIN)
    test = pd.read_csv(utils.RUTA_TEST)
    sabotear(args.escenario, train, test).to_csv(utils.RUTA_TRAIN, index=False)

    log = utils.configurar_log("romper")
    log.warning("Escenario %d aplicado a train.csv: %s", args.escenario, ESCENARIOS[args.escenario])
    log.warning("Ejecute: python run_pipeline.py --sin-preparar")


if __name__ == "__main__":
    main()
