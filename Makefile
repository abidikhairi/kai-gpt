.PHONY: build test install

build:
	python -m build

test:
	python -m pytest
	
install:
	python -m pip install -e .

