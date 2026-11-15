.PHONY: all experiments soft icm figures test

all: experiments soft icm figures

experiments:  ## 12 models x 3 tasks, 5-fold CV, 9 test runs  (~8 min)
	cd src && python run_experiments.py
soft:         ## learning with disagreement for Task 2.1 + soft test run
	cd src && python soft_labels.py
icm:          ## official metric (PyEvALL needs its own venv, see README)
	.venv-pyevall/bin/python src/evaluate_pyevall.py
figures:
	cd src && python eda.py && python make_figures.py && python error_analysis.py
test:
	pytest -q tests && python src/validate_runs.py
