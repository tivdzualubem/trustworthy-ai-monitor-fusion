.PHONY: install verify test reproduce-cpu reproduce-v2-historical

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

verify:
	python scripts/verify_reproducibility.py --strict-hashes

test:
	pytest -q

reproduce-cpu:
	bash scripts/reproduce_cpu_results.sh

reproduce-v2-historical:
	bash scripts/reproduce_historical_v2_evidence.sh
