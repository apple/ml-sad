# define the name of the virtual environment directory
VENV := venv

# default target is to make virtual environment
all: venv

help:
	@echo ""
	@echo ""
	@echo "Makefile commands:"
	@echo ""
	@echo ""
	@echo "make venv - create local Python virtual environment using `requirements-dev.txt` file"
	@echo ""
	@echo ""


$(VENV)/bin/activate: requirements-dev.txt
	python3.7 -m venv $(VENV)
	./$(VENV)/bin/pip install --upgrade pip
	./$(VENV)/bin/pip install -r requirements-dev.txt
	./$(VENV)/bin/pip install rankfm
	./$(VENV)/bin/pip install recommenders
	./$(VENV)/bin/pip install surprise
	./$(VENV)/bin/pip install h5py==2.10.0
	./$(VENV)/bin/pip install cornac

# venv is a shortcut target
venv: $(VENV)/bin/activate

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -depth -name '__pycache__' -exec rm -rf {} ';'

.PHONY: all help venv clean
