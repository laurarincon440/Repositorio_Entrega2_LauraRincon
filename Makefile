# Automatización del pipeline. Uso: make <objetivo>
# En Windows: python run_pipeline.py hace lo mismo que make pipeline.

PYTHON ?= python

.PHONY: help install pipeline prepare validate train evaluate test predict ui clean reproducir romper

help:
	@echo "make install     instala las dependencias de requirements.txt"
	@echo "make pipeline    prepare -> validate -> train -> evaluate -> test"
	@echo "make reproducir  borra resultados previos y ejecuta el pipeline desde cero"
	@echo "make prepare     extrae el dataset y crea la partición 80/20"
	@echo "make validate    valida los datos contra contracts/data_contract.json"
	@echo "make train       entrena la baseline y el modelo seleccionado (MLflow)"
	@echo "make evaluate    valida el modelo con los umbrales de calidad"
	@echo "make test        ejecuta las pruebas con pytest"
	@echo "make predict     predice el ejemplo válido e inválido del contrato"
	@echo "make romper E=3  aplica el sabotaje 3 (1 a 7) y ejecuta el pipeline sin prepare"
	@echo "make ui          abre MLflow UI en http://127.0.0.1:5000"
	@echo "make clean       elimina los resultados generados"

install:
	$(PYTHON) -m pip install -r requirements.txt

# Cada objetivo depende del anterior: si una etapa falla, make se detiene
pipeline: test

prepare:
	$(PYTHON) -m src.prepare

validate: prepare
	$(PYTHON) -m src.validate_data

train: validate
	$(PYTHON) -m src.train

evaluate: train
	$(PYTHON) -m src.evaluate

test: evaluate
	$(PYTHON) -m pytest -q tests

reproducir:
	$(PYTHON) run_pipeline.py --reiniciar

predict:
	$(PYTHON) -m src.predict --ejemplo valida
	$(PYTHON) -m src.predict --ejemplo invalida

# Juego de la Sesión 4: el pipeline debe detenerse en la etapa que detecta el sabotaje
romper:
	$(PYTHON) romper_datos.py $(E)
	$(PYTHON) run_pipeline.py --sin-preparar

ui:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

clean:
	$(PYTHON) -c "import run_pipeline; run_pipeline.reiniciar()"
