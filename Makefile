# Disstruments — local dev
# Prereqs: python 3.11+, node 18+, ffmpeg (`brew install ffmpeg`)
# Pinned to 3.12: best prebuilt-wheel coverage for the ML extras (torch/demucs/panns).
PYTHON ?= python3.12
API_PORT ?= 8642
WEB_PORT ?= 3642

.PHONY: setup setup-ml api web dev test

setup:            ## backend deps (fake-ML capable) + frontend deps
	cd backend && $(PYTHON) -m venv .venv && .venv/bin/pip install -e ".[dev]"
	cd frontend && npm install

setup-ml:         ## real ML models (torch/demucs/panns/librosa) — several GB
	cd backend && .venv/bin/pip install -e ".[ml]"

api:              ## run backend on :$(API_PORT)
	cd backend && .venv/bin/uvicorn "disstruments.main:get_app" --factory --port $(API_PORT)

api-fake:         ## backend with fake ML (no models needed)
	cd backend && DISS_FAKE_ML=1 .venv/bin/uvicorn "disstruments.main:get_app" --factory --port $(API_PORT)

web:              ## run frontend on :$(WEB_PORT)
	cd frontend && DISS_API_PORT=$(API_PORT) npm run dev -- -p $(WEB_PORT)

test:             ## backend test suite (fake ML)
	cd backend && DISS_FAKE_ML=1 .venv/bin/pytest tests -q
