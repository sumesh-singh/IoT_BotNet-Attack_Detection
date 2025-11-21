import os
import threading
import uuid
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Append src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.config_manager import ConfigManager
from data.data_loader import DataLoader
from scripts.train_models import ModelTrainer


app = FastAPI(title="Enhanced IoT BotScan API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


config_manager = ConfigManager('./config/config.yaml')
config = config_manager.config
data_loader = DataLoader(config.get('data', {}))

RUNS: Dict[str, Dict[str, Any]] = {}


@app.get("/api/v1/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/v1/datasets")
def list_datasets() -> Dict[str, Any]:
    paths = config_manager.get('data.data_paths', {}) or {}
    items: List[Dict[str, Any]] = []
    for name, path in paths.items():
        exists = os.path.exists(path)
        item = {"name": name, "path": path, "exists": exists}
        if exists:
            try:
                ds = data_loader.load_dataset(name)
                stats = data_loader.get_dataset_statistics(name)
                item.update({
                    "samples": stats.get('total_samples'),
                    "features": stats.get('n_features'),
                    "classes": stats.get('n_classes'),
                    "label_distribution": stats.get('label_distribution')
                })
            except Exception as e:
                item["error"] = str(e)
        items.append(item)
    return {"items": items, "total": len(items), "page": 1, "page_size": len(items)}


@app.post("/api/v1/datasets/validate")
def validate_dataset(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = payload.get('name')
    if not name:
        raise HTTPException(status_code=422, detail={"code": "invalid_request", "message": "name required"})
    try:
        ds = data_loader.load_dataset(name)
        stats = data_loader.get_dataset_statistics(name)
        return {"name": name, "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=400, detail={"code": "validation_failed", "message": str(e)})


def _run_training_async(run_id: str, datasets: List[str], mode: str) -> None:
    trainer = ModelTrainer('./config/config.yaml')
    try:
        trainer.load_datasets(datasets)
        if mode == 'baseline':
            result = trainer.train_baseline_models(datasets[0])
        elif mode == 'adversarial':
            result = trainer.train_adversarial_robust_models(datasets[0])
        else:
            result = trainer.run_full_training_pipeline(datasets)
        trainer.save_results('./data/results')
        RUNS[run_id].update({
            "status": "completed",
            "finished_at": datetime.now().isoformat(),
            "result": result
        })
    except Exception as e:
        RUNS[run_id].update({
            "status": "failed",
            "finished_at": datetime.now().isoformat(),
            "error": str(e)
        })


@app.post("/api/v1/training/runs")
def create_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    datasets = payload.get('datasets') or ['n_baiot']
    mode = payload.get('mode') or 'baseline'
    run_id = str(uuid.uuid4())
    RUNS[run_id] = {
        "id": run_id,
        "datasets": datasets,
        "mode": mode,
        "status": "running",
        "started_at": datetime.now().isoformat()
    }
    thread = threading.Thread(target=_run_training_async, args=(run_id, datasets, mode), daemon=True)
    thread.start()
    return {"id": run_id, "status": "running"}


@app.get("/api/v1/training/runs")
def list_runs() -> Dict[str, Any]:
    items = list(RUNS.values())
    return {"items": items, "total": len(items), "page": 1, "page_size": len(items)}


@app.get("/api/v1/training/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "run not found"})
    return RUNS[run_id]


@app.get("/api/v1/training/runs/{run_id}/results")
def get_run_results(run_id: str) -> Dict[str, Any]:
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "run not found"})
    run = RUNS[run_id]
    if run.get("status") != "completed":
        raise HTTPException(status_code=202, detail={"code": "not_ready", "message": "results not ready"})
    return {"result": run.get("result")}


@app.get("/api/v1/models")
def list_models() -> Dict[str, Any]:
    models_dir = os.path.join('.', 'data', 'results', 'models')
    items: List[Dict[str, Any]] = []
    if os.path.exists(models_dir):
        for fname in os.listdir(models_dir):
            if fname.endswith('.pkl'):
                items.append({"name": fname, "path": os.path.join(models_dir, fname)})
    return {"items": items, "total": len(items), "page": 1, "page_size": len(items)}


@app.get("/api/v1/reports")
def list_reports() -> Dict[str, Any]:
    results_dir = os.path.join('.', 'data', 'results')
    items: List[Dict[str, Any]] = []
    if os.path.exists(results_dir):
        for fname in os.listdir(results_dir):
            if fname.endswith('.yaml') or fname.endswith('.txt'):
                items.append({"name": fname, "path": os.path.join(results_dir, fname)})
    return {"items": items, "total": len(items), "page": 1, "page_size": len(items)}


@app.get("/api/v1/files")
def get_file(path: str):
    safe_root = os.path.abspath(os.path.join('.', 'data', 'results'))
    full_path = os.path.abspath(path)
    if not full_path.startswith(safe_root):
        raise HTTPException(status_code=403, detail={"code": "forbidden", "message": "invalid path"})
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "file not found"})
    return FileResponse(full_path)


@app.get("/api/v1/drift/events")
def list_drift_events() -> Dict[str, Any]:
    # Placeholder: drift events can be sourced from trainer results YAML
    return {"items": [], "total": 0, "page": 1, "page_size": 0}


@app.get("/api/v1/logs/recent")
def recent_logs() -> Dict[str, Any]:
    logs_dir = os.path.join('.', 'logs')
    entries: List[Dict[str, Any]] = []
    def read_log(fp):
        try:
            with open(fp, 'r', encoding='utf8') as f:
                lines = f.readlines()[-100:]
                return [l.strip() for l in lines]
        except Exception:
            return []
    entries.extend([{ "file": 'iot_botscan.log', "lines": read_log(os.path.join(logs_dir, 'iot_botscan.log')) }])
    entries.extend([{ "file": 'iot_botscan_errors.log', "lines": read_log(os.path.join(logs_dir, 'iot_botscan_errors.log')) }])
    return {"items": entries}
