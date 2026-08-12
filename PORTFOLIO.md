# Federated Health Demo - Portfolio brief (10/10)

## Elevator pitch

Two synthetic hospitals each train a **readmission risk logistic model** on local discharge data. A coordinator runs **FedAvg** on uploaded coefficients only - the step-through UI and audit log prove patient rows never leave site.

## Skills demonstrated

- **Backend:** Python, FastAPI, federated averaging, pure-Python ML
- **Frontend:** Step-through training wizard, privacy audit trail
- **Domain:** Multi-site health data collaboration, PHI boundary education
- **Engineering:** Session state machine, explicit weight-only transmission contract

## Interview line

*"I built a federated learning demo where two hospitals train locally and only share aggregated weights - the UI walks through each site, the federation round, and global metrics while logging that raw data never exported."*

## Run locally in 30 seconds

```powershell
.\run.ps1
# → http://127.0.0.1:8099
```

## Disclaimer

Synthetic data only. Educational federated pattern - not a production FL platform.
