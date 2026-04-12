# Deep-Well AI: Subsurface Inference Engine

Deep-Well AI is a high-performance backend service designed for the energy sector. It provides real-time, automated analysis of borehole data to predict **Enhanced Oil Recovery (EOR)** suitability, **Lithology** classification, and **Subsurface Risk** assessment using a hybrid approach of Machine Learning and Heuristic Physical Validation.

## Technical Architecture
- **Core Framework:** FastAPI (Asynchronous ASGI)
- **Data Orchestration:** Pandas & NumPy (Vectorized Pipeline)
- **Machine Learning:** Scikit-Learn Pipeline (RandomForest/XGBoost artifacts)
- **Concurrency:** Thread-safe singleton pattern for model caching
- **Performance:** O(n) complexity for batch inference

## Installation & Setup

### Prerequisites
- Python 3.12+
- Proxmox or local Linux environment (Recommended: Arch Linux or Ubuntu 22.04+)

### Deployment
1. Clone the repository and navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Setup a localized Python environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Unix/macOS
   .\.venv\Scripts\activate   # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the production-ready server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## API Reference

### Unified Inference Engine
The `predict/unified` endpoint acts as an Intelligent Router. It automatically identifies the analysis context (EOR vs. Lithology) based on the input schema.

- **Endpoint:** `/api/v1/chat`
- **Method:** `POST`
- **Format:** `multipart/form-data`
- **Payload:**
    - `file`: CSV raw well-log data.
    - `model_id`: (Optional) Specify a specific version (e.g., `deepwell-unified-v1`).

### Data Integrity & Physical Gates
To ensure "Mission Critical" reliability, every request passes through a **Physical Validation Layer**:

1. **Sanitization:** Standard industry null values (`-999.25`, `-9999`) are automatically purged to `NaN`.
2. **Auto-Normalization:** If `NPHI` (Neutron Porosity) values exceed `1.0`, the engine treats them as percentages and auto-scales them to decimal format.
3. **Physical Constraints:**
    - **GR (Gamma Ray):** Clamped to `0-300 API` to eliminate electronic noise spikes.
    - **RHOB (Bulk Density):** Values outside `1.0 - 3.5 g/cm³` are invalidated to prevent model hallucinations on physically impossible rock densities.

---

## Response Contract

The backend returns a comprehensive analytical report in JSON format:

```json
{
  "id": "dw-unique-id",
  "model_used": "deepwell-unified-v1",
  "detection_type": "lithology",
  "warnings": [
    "INFO: NPHI auto-scaled to decimal.",
    "WARNING: 5 rows suppressed due to RHOB physical violations."
  ],
  "predictions": [
    {
      "depth": 5500.0,
      "lithology": "Sandstone",
      "risk_score": 0.15,
      "confidence": 0.92,
      "anomaly_detected": false
    }
  ]
}
```

---

## Testing with cURL

You can test the API directly from your terminal using the following command. Replace `test_data.csv` with the path to your actual file.

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/predict/unified' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test_data.csv;type=text/csv' \
  -F 'model_id=deepwell-unified-v1'
```

## Directory Structure
- `/app/main.py`: Entry point and FastAPI route definitions.
- `/app/services/inference.py`: Core Engine (Sanitization, Physics Validation, ML Inference).
- `/app/models/`: Directory for ML artifacts (`.pkl`).
- `/app/schemas.py`: Pydantic models for I/O validation.
