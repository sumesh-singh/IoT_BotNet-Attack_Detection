# Enhanced IoT BotScan

Comprehensive IoT botnet detection system with real-time updates and hybrid ensemble learning.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- SQLite / PostgreSQL (optional, for persistent storage)

### 1. Installation

1.  **Clone/Navigate to the directory**:
    ```bash
    cd enhanced_iot_botscan
    ```

2.  **Install Dependencies**:
    > **Important**: Ensure you install all requirements, including `strawberry-graphql` which is required for the API.
    ```bash
    pip install -r requirements.txt
    ```

### 2. Configuration

1.  **Environment Variables**:
    Copy the example environment file and adjust if necessary.
    ```bash
    cp .env.example .env
    # On Windows: copy .env.example .env
    ```

2.  **Data Paths**:
    Ensure your `config/config.yaml` (main config) points to valid data directories for `n_baiot`, `iot_23`, and `bot_iot`.

### 3. Running the Project

#### 🌐 Backend API Server
Starts the FastApi server with GraphQL and WebSocket support.
```bash
# Run as a module from the project root
python -m src.api.main
```
- API Docs: `http://localhost:8000/docs`
- GraphQL Playground: `http://localhost:8000/graphql`
- Dashboard: `http://localhost:8000/dashboard`

#### 🖥️ Streamlit Dashboard
Starts the interactive data science frontend.
```bash
streamlit run app.py
```
- Access at: `http://localhost:8501`

### 4. Running Verification & Benchmarks

We have included scripts to verify compliance with project requirements.

```bash
# Validate Multi-Dataset Support (REQ-013)
python tests/validate_multidataset.py

# Benchmark Performance (REQ-003, REQ-021)
python tests/benchmark_performance.py

# Validate Adversarial Robustness (REQ-006)
python tests/validate_adversarial.py
```

## 📂 Project Structure
- `src/api`: Backend API logic (FastAPI, GraphQL, WebSockets).
- `src/core`: Core ML logic (Ensemble models, Drift detection).
- `src/streamlit_app`: Frontend logic.
- `tests/`: Verification and benchmark scripts.
- `web/`: HTML templates for API dashboard.
