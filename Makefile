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
	$(PYTHON) scripts/01_data_exploration.py
	$(PYTHON) scripts/02_clean_data.py
# 	$(PYTHON) scripts/03_od_analysis.py
# 	$(PYTHON) scripts/04_station_segmentation.py
# 	$(PYTHON) scripts/05_demand_prediction.py

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:
	rm -rf data/processed figures models

.PHONY: install pipeline clean
