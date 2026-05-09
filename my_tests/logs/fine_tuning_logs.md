# Fine-Tuning Experiment Logs: Domain Adaptation for ReFinED

This log documents the fine-tuning experiments conducted for Chapter 6. The objective was to evaluate the impact of domain-specific and merged training data on entity linking performance for the *Movies* and *Companies* domains.

## 1. Experimental Environment
- **GPU:** NVIDIA GeForce RTX 4070 Ti (System B)
- **Base Model:** ReFinED 
- **Entity set:** Wikidata 
- **Batch Size:** Physical Batch Size: 128

---

## 2. Model Performance Summary
*Evaluation Mode: Row-context inference. Metrics represent entity-level performance.*

| Model ID           | Training Rows | Duration | Movies (F1) | Companies (F1) |
|:-------------------|:--------------|:---------|:------------|:---------------|
| **merged_10k**     | 10,000        | ~15 min  | 0.9253      | 0.8592         |
| **merged_60k**     | 60,000        | 1h 37m   | 0.9335      | 0.8673         |
| **merged_full**    | 135,000       | 3h 13m   | **0.9414**  | 0.8772         |
| **companies_full** | 67,500        | 1h 34m   | 0.5222      | **0.8783**     |
| **movies_full**    | 67,500        | 1h 21m   | 0.9390      | 0.1451         |

---

## 3. Detailed Per-Model Metrics

### merged_10k
*Mixed domain training (5k rows Movies / 5k rows Companies)*

| Dataset       | Precision | Recall |  F1-Score  |
|:--------------|:---------:|:------:|:----------:|
| **Movies**    |  0.9683   | 0.8860 | **0.9253** |
| **Companies** |  0.9639   | 0.7750 | **0.8592** |

### merged_60k
*Mixed domain training (30k rows Movies / 30k rows Companies)*

| Dataset       | Precision | Recall |  F1-Score  |
|:--------------|:---------:|:------:|:----------:|
| **Movies**    |  0.9791   | 0.8920 | **0.9335** |
| **Companies** |  0.9798   | 0.7780 | **0.8673** |

### merged_full
*Mixed domain training (67.5k rows Movies / 67.5k rows Companies)*

| Dataset       | Precision | Recall |  F1-Score  |
|:--------------|:---------:|:------:|:----------:|
| **Movies**    |  0.9966   | 0.8921 | **0.9414** |
| **Companies** |  0.9924   | 0.7868 | **0.8772** |

### companies_full
*Single domain training (67.5k rows Companies)*

| Dataset       | Precision | Recall |  F1-Score  |
|:--------------|:---------:|:------:|:----------:|
| **Movies**    |  0.8201   | 0.3830 | **0.5222** |
| **Companies** |  0.9887   | 0.7900 | **0.8783** |

### movies_full
*Single domain training (67.5k rows Movies)*

| Dataset       | Precision | Recall |  F1-Score  |
|:--------------|:---------:|:------:|:----------:|
| **Movies**    |  0.9900   | 0.8930 | **0.9390** |
| **Companies** |  0.8876   | 0.0792 | **0.1451** |

---