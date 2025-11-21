## Overview
Build a fully functional application without authentication, integrating the React frontend with the existing Python backend. Use RESTful APIs (JSON) with optional SSE for live updates. Ensure consistent data schemas, robust error handling, comprehensive testing, and production-ready deployment with CORS and monitoring.

## Frontend–Backend Integration
### API Design
- Base URL: `/api/v1` (versioned)
- Content-Type: `application/json` for requests and responses
- Endpoints (initial set):
  - `GET /datasets` – list datasets with stats (name, samples, features, classes)
  - `POST /datasets/validate` – validate dataset path or uploaded file and return issues/stats
  - `POST /training/runs` – create a training run (datasets, mode, config, seeds); returns `run_id`
  - `GET /training/runs` – list past runs (status, metrics, timestamps)
  - `GET /training/runs/{run_id}` – status/progress, current phase, metrics snapshot
  - `GET /training/runs/{run_id}/results` – final metrics and artifact links (YAML, model .pkl)
  - `GET /models` – list saved models and metadata (created_at, dataset, params)
  - `GET /reports` – list evaluation reports and plots; filter by dataset/run
  - `GET /drift/events` – list drift detections and statistics
  - `GET /logs/recent` – latest log entries with levels and timestamps
- Live updates (optional): `GET /training/runs/{run_id}/events` via SSE (text/event-stream) for progress/metric updates
- Artifacts: serve files from `./data/results` via `/api/v1/files/{path}` with whitelist to prevent path traversal

### Data Schemas (examples)
- `TrainingRun`: `{ id, datasets[], mode: 'baseline'|'adversarial'|'full', status: 'queued'|'running'|'completed'|'failed', metrics: {accuracy, precision, recall, f1, roc_auc}, started_at, finished_at }`
- `DatasetInfo`: `{ name, samples, features, classes, label_distribution[] }`
- `ModelInfo`: `{ id, type, dataset, params, created_at, path }`
- `Error`: `{ code, message, details? }`

### Frontend Integration
- API client: Axios instance with base URL env `VITE_API_BASE_URL`
- Server state: TanStack Query (caching, retries, stale times)
- Navigation: React Router routes mapped to modules (Datasets, Training, Evaluation, Models, Drift, Logs, Settings)
- Live run status: SSE or polling fallback with exponential backoff

### REST Conventions
- Use nouns for resources; plural collections; proper HTTP verbs
- Status codes: 200/201 success, 202 accepted (async run start), 400/422 validation errors, 404 not found, 500 internal
- Errors standardized with `{code, message, details}`; frontend displays toast and inline messages

## Functional Requirements (No-Auth)
- Core features fully accessible:
  - Browse datasets and stats, validate dataset
  - Configure and start training runs; view live progress; view results and artifacts
  - View models and evaluation reports (confusion matrix, ROC)
  - Inspect drift detections and logs
- Error handling:
  - Frontend: global Axios interceptor for errors; UI toasts and per-view messages
  - Backend: try/except with structured logging; map exceptions to 4xx/5xx with clear messages
- Consistent formats:
  - All lists paginated with `{items, total, page, page_size}`; timestamps ISO 8601

## Testing Protocol
- Unit tests (frontend): component rendering and utilities; mock API
- Integration tests (frontend): Query hooks and pages with mocked endpoints
- E2E tests (Playwright):
  - Dataset browsing → run start → progress view → results view → artifact download
  - Reports view and chart rendering; drift detections page
- Backend tests (pytest): endpoint contract tests, error scenarios, SSE stream
- Cross-browser/device: Chrome/Edge/Firefox; responsive checks at 1280/1024/768/480 using Playwright device emulation

## Deployment Checklist
- Environment variables:
  - Backend: `API_PORT`, `RESULTS_DIR`, `CORS_ALLOWED_ORIGINS`, `LOG_LEVEL`
  - Frontend: `VITE_API_BASE_URL`
- CORS:
  - Allow frontend origin, methods (GET/POST), headers (Content-Type), credentials disabled
- Monitoring:
  - Backend: request timing and error rate; expose `/api/v1/health`
  - Frontend: Web Vitals collection; console error aggregation (optional)
- Documentation:
  - Endpoint reference with request/response examples
  - Data schemas and error formats
  - Integration flows and artifact paths

## Implementation Strategy & Milestones
1. API layer scaffolding (controllers for datasets, training runs, models, reports, drift, logs); file serving with whitelist
2. Frontend API client and hooks; pages wired to endpoints; SSE/polling for run progress
3. Error handling: standardized backend errors; frontend interceptors and UI messaging
4. Testing: unit/integration; E2E flows; cross-browser checks
5. Deployment: env configs; CORS; health endpoint; basic monitoring; build and serve frontend behind `/`

## Performance Optimization
- Frontend: route-level code splitting, query caching, virtualization for large tables
- Backend: chunked file reads, pagination, gzip responses

## Deliverables
- Fully functional app with integrated frontend and backend (no auth)
- REST API documentation and data schemas
- Test suite: unit/integration/E2E with reports
- Deployment scripts/config and monitoring hooks
- Developer and user documentation