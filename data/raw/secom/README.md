# UCI SECOM 原始資料放置說明

本資料夾用來放置 WaferWatch Phase R1 使用的 UCI SECOM 原始資料。

## 已放置的必要檔案

本階段需要以下三個檔案：

```text
data/raw/secom/secom.data
data/raw/secom/secom_labels.data
data/raw/secom/secom.names

```

---

## 本階段用途

UCI SECOM dataset 會作為 WaferWatch 的公開半導體製造感測器資料骨幹，用來示範：

- 原始資料匯入
- 缺失值檢查
- 類別不平衡檢查
- 清理後資料輸出

---

## Phase R1 預期執行指令

```powershell
python -m src.data.ingest_secom
python -m src.data.validate_secom
python -m src.data.clean_secom
```

---

## Phase R1 預期輸出

```text
data/processed/secom_cleaned.csv
reports/secom_data_quality_report.md
reports/secom_class_imbalance_report.md
```

---

## 階段範圍限制

本階段只處理 UCI SECOM dataset，不做 RAG、不做 Docker、不做 GitHub 部署。
