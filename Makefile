.PHONY: setup lab docker test
setup:      ## venv + editable install
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[notebook]"
lab:        ## open the notebook locally
	. .venv/bin/activate && jupyter lab notebooks/summarize.ipynb
docker:     ## run in Docker (Ollama on host)
	docker compose -f docker/docker-compose.yml up --build
test:
	. .venv/bin/activate && python -m pytest -q tests
