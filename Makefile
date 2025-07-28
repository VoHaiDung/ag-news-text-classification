.PHONY: setup install clean test run-app

setup:
	python -m venv venv
	. venv/bin/activate && pip install -e .

install:
	pip install -r requirements.txt

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

test:
	pytest tests/

run-app:
	streamlit run app/streamlit_app.py

download-data:
	python scripts/setup/download_all_data.py

prepare-data:
	python scripts/data_preparation/prepare_ag_news.py

train-baseline:
	python experiments/baselines/neural/bert_vanilla.py
