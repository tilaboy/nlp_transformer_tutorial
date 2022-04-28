pip := pip3.9
python := python3.9
MODULE_FOLDER := fork_python_repo
MODULE_VERSION := $(if $(CI_COMMIT_TAG),$(CI_COMMIT_TAG),0.0.0)

poetry.lock: pyproject.toml
	$(pip) install --upgrade pip
	command -v poetry || $(pip) install poetry
	poetry env use $(python)
	poetry update
	touch poetry.lock

deps: poetry.lock

install: deps
	poetry install

test: install
	poetry run pytest -vv -o log_cli=true --cov=$(MODULE_FOLDER) tests

analysis: deps
	poetry run flake8 $(MODULE_FOLDER)
	poetry run flake8 tests

tidy: deps
	poetry run black --experimental-string-processing $(MODULE_FOLDER)
	poetry run black --experimental-string-processing tests

dist: deps
	sed -i "s/version = \"0.0.0\"/version = \"$(MODULE_VERSION)\"/g" pyproject.toml
	poetry version $(MODULE_VERSION)
	poetry build -f wheel

docs: deps
	poetry run $(pip) install pdoc3
	poetry run pdoc --output-dir docs $(MODULE_FOLDER)
	# use this command if you want to check the html version
	# poetry run pdoc --http localhost:8081 $(MODULE_FOLDER)

hook: deps
	poetry run pre-commit install

clean:
	find . \( -name "*.py[co]" -or -name dist -or -name .pytest_cache -or -name __pycache__ -or -name "*.egg-info" -or -name .eggs -or -name build -or -name poetry.lock \) -exec rm -rf {} +

nuke: clean
	poetry env remove $(python) -q || true
	printf "\n!!! Warning: you no longer have a virtual environment for your pre-commit hook, re-run 'make hook'\n"

.PHONY: deps test analysis tidy install dist docs hook copyright license clean nuke
