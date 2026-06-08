PYTHON = .venv/bin/python

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

.venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

install: .venv

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

data/processed figures models:
	mkdir -p $@

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

pipeline: install data/processed figures models
# 	$(PYTHON) scripts/02_data_raw_plots.py
	$(PYTHON) scripts/02_data_clean.py
	$(PYTHON) scripts/02_data_clean_plots.py
# 	$(PYTHON) scripts/03_feature_engineering_od_analysis.py
# 	$(PYTHON) scripts/03_feature_engineering_stations.py
# 	$(PYTHON) scripts/04_model_baseline.py
# 	$(PYTHON) scripts/05_model_segmentation.py
# 	$(PYTHON) scripts/05_model_prediction.py
# 	$(PYTHON) scripts/06_evaluation.py
# 	$(PYTHON) scripts/07_decision_layer.py

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:
	rm -rf data/processed figures models

.PHONY: install pipeline clean
