# Disstruments — local dev
# Prereqs: python 3.11+, node 18+, ffmpeg (`brew install ffmpeg`)

.PHONY: setup setup-ml api web dev test

setup:            ## backend deps (fake-ML capable) + frontend deps
	cd backend && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
	cd frontend && npm install

setup-ml:         ## real ML models (torch/demucs/panns/librosa) — several GB
	cd backend && .venv/bin/pip install -e ".[ml]"

api:              ## run backend on :8000
	cd backend && .venv/bin/uvicorn "disstruments.main:get_app" --factory --port 8000

api-fake:         ## backend with fake ML (no models needed)
	cd backend && DISS_FAKE_ML=1 .venv/bin/uvicorn "disstruments.main:get_app" --factory --port 8000

web:              ## run frontend on :3000
	cd frontend && npm run dev

test:             ## backend test suite (fake ML)
	cd backend && DISS_FAKE_ML=1 .venv/bin/pytest tests -q
