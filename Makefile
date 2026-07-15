.PHONY: install verify test reproduce-cpu

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

verify:
	python scripts/verify_reproducibility.py --strict-hashes

test:
	pytest -q

reproduce-cpu:
	bash scripts/reproduce_cpu_results.sh
