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
	pip install pytest black isort

build-cython:
	python setup.py build_ext -i --force

build:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python setup.py build_ext -i --force
	python setup.py install
