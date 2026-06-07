# Convenience targets for the AG News capstone pipeline.
# Each target maps to one phase of the WBS so a fresh checkout can be
# reproduced with a single ``make all``.

PYTHON ?= python

.PHONY: all install lint test phase1 phase2 phase3 phase4 phase5 phase6 phase7 phase8 phase9 clean

install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src scripts tests
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest tests

phase1:
	$(PYTHON) -m scripts.phase1_kickoff --config configs/base.yaml

phase2:
	$(PYTHON) -m scripts.phase2_eda --config configs/data/ag_news.yaml

phase3:
	$(PYTHON) -m scripts.phase3_baselines --config configs/data/ag_news.yaml

phase4:
	$(PYTHON) -m scripts.phase4_transformers \
		--data-config configs/data/ag_news.yaml \
		--model-configs configs/models/deberta_v3_small.yaml configs/models/modernbert_base.yaml

phase5:
	$(PYTHON) -m scripts.phase5_multilingual \
		--source-data-config configs/data/ag_news.yaml \
		--target-data-config configs/data/ag_news_vi.yaml \
		--model-config configs/models/mdeberta_v3.yaml

phase6:
	$(PYTHON) -m scripts.phase6_setfit \
		--data-config configs/data/ag_news.yaml \
		--model-config configs/models/setfit.yaml

phase7:
	@echo "Phase 7 requires --model-dir, --predictions-csv and --probabilities-npy paths."
	@echo "Example: make phase7 ARGS='--model-dir outputs/transformers/.../best ...'"
	$(PYTHON) -m scripts.phase7_evaluation $(ARGS)

phase8:
	@echo "Phase 8 requires --model-dir."
	$(PYTHON) -m scripts.phase8_deployment $(ARGS)

phase9:
	$(PYTHON) -m scripts.phase9_report

all: phase1 phase2 phase3 phase4 phase5 phase6 phase9
	@echo "Phase 7 and 8 are interactive: run them after the model checkpoints exist."

clean:
	rm -rf outputs build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
