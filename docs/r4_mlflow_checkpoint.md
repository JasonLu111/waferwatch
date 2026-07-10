# WaferWatch Phase R4 Checkpoint: MLflow Tracking and Registry

Checkpoint date: 2026-07-10

## Scope

This checkpoint covers only the first part of Phase R4 MLOps:

- MLflow experiment tracking
- Local SQLite tracking backend
- Local artifact storage
- Model Registry
- Champion model alias

FastAPI, Docker, monitoring integration, GitHub Actions, and the final
system architecture document are not completed yet.

## Completed

- Added `src/mlops/register_models.py`.
- Added `src/mlops/__init__.py`.
- Added `scripts/start_mlflow.ps1`.
- Added local MLflow runtime paths to `.gitignore`.
- Started MLflow 3.14.0 using a SQLite backend.
- Logged the deployment-eligible supervised models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Selected Logistic Regression as the deterministic champion because all
  compared deployment metrics were tied and it had the smallest artifact.
- Registered `WaferWatchRiskModel`.
- Assigned the `champion` alias to model version 1.
- Loaded `models:/WaferWatchRiskModel@champion` as a scikit-learn Pipeline.
- Successfully generated model risk probabilities.

## Known Unresolved Issue

The temporary registry verification successfully loaded the champion model
and generated predictions, but its final assertion failed because:

- The Run ID resolved through the `champion` registry alias
- The Run ID stored in `mlflow_data/registration_summary.json`

did not match.

Observed registry alias Run ID:

`d43601bddeae47c4a06ce258d6c599bd`

This is currently treated as a metadata consistency issue rather than a
model-loading failure.

Do not rerun model registration before inspecting this mismatch, because
another registration run could create additional model versions and make the
diagnosis less clear.

## Local Runtime State

The following path is intentionally excluded from Git:

`mlflow_data/`

It contains:

- `mlflow.db`
- MLflow artifacts
- `registration_summary.json`

This local directory should be preserved for the next session.

## Next Session

Continue in this order:

1. Start the local MLflow server.
2. Read the registry alias Run ID.
3. Read the registration summary Run ID.
4. determine why the IDs differ.
5. Make registration and verification deterministic and repeatable.
6. Rerun the registry verification.
7. Begin FastAPI only after registry verification passes.

## Restart Command

Terminal 1:

```powershell
Set-Location D:\waferwatch
.\.venv\Scripts\Activate.ps1
& .\scripts\start_mlflow.ps1
Do not rerun python -m src.mlops.register_models until the existing
Run ID mismatch has been inspected.
