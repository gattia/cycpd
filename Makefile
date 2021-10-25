autoformat:
	set -e
	isort .
	black --config pyproject.toml .

lint:
	set -e
	isort -c .
	black --check --config pyproject.toml .

test:
	set -e
	pytest

dev:
	pip install pytest black isort twine wheel

requirements:
	python -m pip install -r requirements.txt

build-cython:
	python setup.py build_ext -i --force

build:
	python setup.py build_ext -i --force
	python setup.py install
