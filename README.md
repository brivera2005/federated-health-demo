# Federated Health Demo

**Project 10 of 10** in the healthcare portfolio series - two-hospital federated readmission risk training with weights-only sync.

![Python](https://img.shields.io/badge/Python-3.11+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

## What it does

- **Two fake hospitals** (Riverview & Summit) train a logistic readmission model on local synthetic cohorts
- **Federated averaging** combines sample-weighted coefficients at a coordinator
- **Step-through UI**: train Site A → train Site B → federated round → global metrics
- **Privacy audit log** confirms `raw_data_exported=false` - weights never include PHI

> **Demo only** - synthetic patients, educational FL pattern, not for clinical use.

## Quick start (Windows)

```powershell
cd C:\Users\brive\Projects\federated-health-demo
.\run.ps1
```

Open **http://127.0.0.1:8099**

## API

| Endpoint | Description |
|----------|-------------|
| `GET /` | Step-through UI |
| `GET /api/health` | Health check |
| `GET /api/state` | Current demo state |
| `POST /api/step` | Advance one step |
| `POST /api/reset` | Reset session |

## Project structure

```
app/
 main.py FastAPI routes
 federated.py Session + FedAvg orchestration
 ml.py Logistic model + synthetic data
 models.py Pydantic models
static/ Step-through UI
```

See **[PORTFOLIO.md](./PORTFOLIO.md)** for interview talking points.

## License

MIT - see [LICENSE](./LICENSE)
