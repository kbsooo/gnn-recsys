# Implementation Plan for `g_recsys_*` Notebooks

This plan consolidates the strategies described in `docs/claude_v2_gnn.md`, `docs/claude_v2_dl.md`, `docs/gpt_v2.md`, and the project requirements in `README.md`. It defines the pipeline that both notebooks (`g_recsys_gnn_v1.ipynb`, `g_recsys_dl_v1.ipynb`) will follow.

---

## 1. Objectives Recap

- **Primary deliverables**: two notebooks implementing the README workflow with (1) a LightGCN-based recommender (`g_recsys_gnn_v1.ipynb`) and (2) a Non-GNN deep learning recommender (`g_recsys_dl_v1.ipynb`).
- **Task requirements** (README): analyze the dataset, preprocess appropriately, build/justify models, train on train.csv, evaluate with suitable metrics, and provide an inference routine that emits `user item recommend` tuples plus a `Total recommends = ...` summary.
- **Strategy alignment**:
  - Follow `docs/claude_v2_gnn.md` for LightGCN + BPR loss, negative sampling, and Recall@K/NDCG@K evaluation.
  - Follow `docs/claude_v2_dl.md` for NeuMF/NCF-style architecture, BPR/BCE losses, and threshold-based inferencing.
  - Use `docs/gpt_v2.md` unified recommendations for consistent thresholds (tune ≥3.5 vs ≥4.0) and baseline comparisons.

---

## 2. Shared Data & Experiment Pipeline

1. **Load & Inspect** (`dataset 분석`):
   - Read `data/train.csv`.
   - Report counts of users, items, interactions, sparsity, per-user/ per-item degree stats, and rating histogram (consistent with docs).
2. **ID Mapping**:
   - Map raw user/item IDs to contiguous indices for tensor operations.
   - Persist mapping dictionaries (or functions) for inference.
3. **Label Definition**:
   - Create binary preference labels using an adjustable threshold `rating >= t`.
   - Start with `t=4.0`, but expose `t` as a tunable hyperparameter so both thresholds (3.5/4.0) can be validated as suggested in `docs/gpt_v2.md`.
4. **Split Strategy** (per docs and README requirement for planfulness):
   - User-stratified 80/10/10 random split ensuring each user has ≥1 interaction in every subset when possible.
   - Optionally fall back to leave-one-out if a user has very few interactions.
5. **Negative Sampling**:
   - For implicit-feedback training loops (both models), sample `num_negatives=4` unseen items per positive (aligned with `claude_v2_gnn` & `claude_v2_dl` suggestions).
   - Provide helper to regenerate negatives each epoch if desired.
6. **Evaluation Metrics**:
   - Offline: Recall@K, NDCG@K (K=10/20), HitRate@K, coverage.
   - Binary threshold tuning: accuracy, precision, recall, F1 on validation to pick a score cutoff for `O/X`.
7. **Inference Output**:
   - Standardized function `predict_ox(test_df)` used by both notebooks to emit formatted table and totals (README requirement).

This shared pipeline will be implemented as reusable cells (data prep, sampling, evaluation) that both notebooks can import via `%run` or by copy-sharing code blocks (since the deliverable is two self-contained notebooks).

---

## 3. Notebook Structure Blueprint

Each notebook will follow the same major sections to satisfy the README submission order:

1. **Title + Overview** (with references to strategy docs).
2. **Dataset Analysis & Preprocessing**:
   - EDA tables/plots.
   - Split and mapping logic.
3. **Model Section**:
   - Architecture diagram/description.
   - Hyperparameters table.
4. **Training & Metrics**:
   - Loss curves (loss function 그래프 requirement).
   - Validation metrics table.
5. **Inference Demo**:
   - Mock `test_df` example (or placeholder instructions) showing final `O/X` output format.
6. **Reflection**:
   - Lessons / unique ideas, consistent with README section 4.

---

## 4. LightGCN Notebook Plan (`g_recsys_gnn_v1.ipynb`)

1. **Modeling Choices**:
   - Adopt the LightGCN config from `docs/claude_v2_gnn.md`: embedding size 64, 3 layers, dropout 0.1, Adam lr=1e-3, weight decay=1e-4.
   - Use BPR loss with sampled negatives.
   - Support optional popularity prior for degree-normalized initialization, per strategy doc.
2. **Implementation Outline**:
   - Build sparse adjacency matrix (PyTorch sparse COO) with symmetric normalization.
   - Propagate embeddings via simplified convolution (no feature transform).
   - Score function = dot product of user/item embeddings.
3. **Training Loop**:
   - Mini-batch sampling of positive edges + corresponding negatives.
   - Track epoch loss and validation metrics every few epochs.
4. **Evaluation & Thresholding**:
   - Compute Recall@10/20 and NDCG@10/20 on validation.
   - Grid-search threshold for final `O/X` inference.
5. **Inference Cell**:
   - Function to accept arbitrary `(user,item)` CSV path or DataFrame and output formatted table + totals.

---

## 5. Non-GNN Notebook Plan (`g_recsys_dl_v1.ipynb`)

1. **Modeling Choices**:
   - Base architecture: NeuMF variant from `docs/claude_v2_dl.md` (GMF + MLP fusion).
   - Embedding dims: 64 for both user & item; MLP hidden sizes [128, 64, 32, 16] with dropout=0.2.
   - Loss: Binary cross-entropy with logits (for predicted preference) OR BPR; start with BCE for stability, optionally mention BPR variant.
2. **Implementation Outline**:
   - DataLoader that yields `(user, pos_item, neg_items)` similarly to GNN pipeline for fairness.
   - Optionally include side features (rating normalization) as described in doc, but keep baseline simple.
3. **Training Loop**:
   - Use Adam lr=1e-3, weight decay=1e-5, early stopping on validation Recall@10.
   - Plot BCE loss curve.
4. **Evaluation & Thresholding**:
   - Same evaluation helpers as LightGCN to enable direct comparison.
   - Determine inference threshold using validation ROC-style sweep.
5. **Inference Cell**:
   - Reuse shared inference routine; ensure consistent formatting.

---

## 6. Risk Mitigation & Validation

- **Cold-start handling**: fallback to popularity score (global hit rate) when user/item unseen. Documented in both notebooks per doc recommendations.
- **Reproducibility**: set manual seeds, log hyperparameters, and store mapping dictionaries to ensure inference consistency.
- **Testing hooks**: include lightweight sanity checks (e.g., verifying positive/negative samples do not overlap) to catch data issues early.

---

## 7. Next Steps

1. Implement shared utilities (EDA, mapping, split, evaluation).
2. Build LightGCN notebook following Section 4.
3. Build NeuMF notebook following Section 5.
4. Run quick smoke tests (small epochs) to verify cells execute.
5. Document lessons learned per README submission order.

This plan ensures both notebooks adhere to the strategic guidance in `/docs` and satisfy the README deliverables.
