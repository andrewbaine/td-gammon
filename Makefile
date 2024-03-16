
ENV_DIR=.venv

$(ENV_DIR)/bin/activate: requirements.txt
	python3 -m venv $(ENV_DIR)
	$(ENV_DIR)/bin/pip install -r requirements.txt

