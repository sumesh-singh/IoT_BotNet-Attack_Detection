## Goal
Deliver a fully working app (no authentication) with React frontend integrated to a Python backend via REST APIs, optional SSE for live training updates, consistent schemas, robust error handling, comprehensive tests, and production‑ready configs (CORS, env, monitoring).

## Backend (Python)
- Stack: FastAPI + Uvicorn
- Versioned base path: `/api/v1`
- Endpoints:
  - Datasets: `GET /datasets`, `POST /datasets/validate`
  - Training: `POST /training/runs`, `GET /training/runs`, `GET /training/runs/{id}`, `GET /training/runs/{id}/results`, optional `GET /training/runs/{id}/events` (SSE)
  - Models: `GET /models`
  - Reports: `GET /reports`
  - Drift: `GET /drift/events`
  - Logs: `GET /logs/recent`
  - Files: `GET /files/{path}` with whitelist
  - Health: `GET /health`
- Implementation notes:
  - Use existing `ModelTrainer`, `DataLoader` to process datasets/runs
  - Run registry: in‑memory map `{id -> status, metrics, timestamps}`; background threads for long runs
  - Error format: `{code, message, details}` with proper HTTP status codes
  - CORS: allow frontend origin, GET/POST, Content‑Type

## Frontend (React + TS)
- API client: Axios with base URL `VITE_API_BASE_URL` (defaults to `/api/v1`)
- TanStack Query hooks per endpoint, polling/SSE for run updates
- Pages wiring:
  - Datasets: list + validate
  - Training: form to start runs + live progress
  - Evaluation: show metrics, confusion matrix, ROC/PR charts from artifacts
  - Models/Reports: list artifacts, download
  - Drift/Logs: list events

## Testing
- Backend: pytest for endpoint contracts, error cases, health
- Frontend: unit/integration tests with mocked APIs; E2E (Playwright) for full flows
- Cross‑browser/device checks (Chrome/Edge/Firefox, common breakpoints)

## Deployment
- Env vars: `API_PORT`, `RESULTS_DIR`, `CORS_ALLOWED_ORIGINS`, `LOG_LEVEL`, `VITE_API_BASE_URL`
- Monitoring: basic request timing/error rate; health endpoint
- Build frontend and serve behind `/` or via separate static hosting with CORS

## Milestones
1) Backend scaffolding & minimal endpoints
2) Frontend hooks & page wiring
3) Error handling & standardized responses
4) Tests (backend + frontend) and E2E flows
5) Deployment configs (env, CORS, health) and basic monitoring

## Acceptance
- All endpoints respond with consistent JSON schemas
- Frontend pages function end‑to‑end without authentication
- Artifacts and reports accessible from UI
- Tests pass and app builds for production