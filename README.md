# Freight Cost Watcher

A reproducible AI pipeline that monitors weekly road freight costs across shipping corridors, flags acute spikes and sustained cost creep, and verifies candidate anomalies against operational context notes using hybrid retrieval and deterministic guardrails.

---

## 1. Architecture & Pipeline Flow

1. **Ingestion & Validation (`src/pipeline/ingestion.py`)**: Validates schema, positive quantities/distances, unique `shipment_id` values, consistent `route_type` per corridor, and maps `shipment_date` to its Monday `week_of` (`YYYY-MM-DD`). Aggregates 2,940 shipments into 728 route-weeks across 7 corridors.
2. **Baseline Engine (`src/pipeline/baselines.py`)**:
   - **Own History**: Mean over a calendar-bounded per-route `deque` of strictly prior Mondays (up to 8 weeks / 56 calendar days, $O(8)$ time per row, recomputed per week to avoid floating-point drift).
   - **Peer Baseline**: Leave-one-out mean across routes of the same `route_type` in the same week, computed in $O(1)$ time via `(bucket_sum - self_cost) / (bucket_count - 1)`.
3. **Hysteresis Anomaly Detector (`src/pipeline/anomaly_detector.py`)**: Flags acute spikes (`vs_own_pct >= 10.0%`) and gradual cost creep (`vs_peer_pct >= 20.0%` and `vs_own_pct >= 5.0%`, sustaining while `vs_peer_pct >= 15.0%` and `vs_own_pct > 0.0%`). Prunes 706 normal route-weeks (97.0%) and forwards 22 candidate anomalies.
4. **Hybrid RAG & Deterministic Guardrails (`src/rag/`, `src/guardrails.py`)**:
   - Note enrichment extracts effective date windows, causal direction (`increase`, `decrease`, `none`), and stated magnitude bounds once per run.
   - Retrieval guarantees inclusion of all notes matching `route` or `"All Routes"`, ranked by date proximity to `week_of`, route specificity, and Chroma ONNX MiniLM dense similarity.
   - Deterministic guardrails verify every LLM justification against 5 strict rules before emitting `FinalOutputRow`.

---

## 2. Core Mathematical Definitions & Edge Cases

- **Group Ratio-of-Sums**: Weekly unit cost is computed as `sum(freight_cost_inr) / sum(quantity_tonnes * distance_km)`. Computing the mean of individual shipment ratios would overweight small loads.
- **Calendar Window vs. Observed Rows**: The own-history window is calendar-based (strictly prior Mondays within 56 days, at most 8, using all available when fewer than 8 exist, with no synthetic padding). Because all 7 routes have shipments in all 104 weeks, this matches `shift(1).rolling(8, min_periods=1).mean()`, verified in `tests/test_baselines.py`.
- **First-Week Behavior**: A route's first week has `prior_weeks_count = 0` and `vs_own_pct = None` (rendered as `"N/A (no prior history)"`), so neither the acute spike rule nor the creep rule can fire on unobserved history.

### Threshold Sensitivity Analysis

Generated directly from `AnomalyDetector.run_sensitivity_analysis()` across the 728 route-weeks:

| Spike Threshold | Creep Peer Entry | Min Own Rise | Hysteresis Keep | Candidate Pool | Behavior |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **10.0%** | **20.0%** | **5.0%** | **15.0%** | **22 rows** | **Baseline Configuration** |
| 8.0% | 20.0% | 5.0% | 15.0% | 23 rows | +1 marginal acute week |
| 12.0% | 20.0% | 5.0% | 15.0% | 21 rows | -1 marginal acute week |
| 10.0% | 18.0% | 4.0% | 12.0% | 22 rows | Identical candidate set |
| 10.0% | 22.0% | 6.0% | 15.0% | 19 rows | Stricter creep entry (-3 weeks) |

---

## 3. Anti-Hallucination Guardrail Rules & Deliberate Decisions

Every `"justified"` verdict from the LLM must pass `verify_verdict_guardrails()` in `src/guardrails.py`:

1. **Route Scope**: `note.applies_to` must equal `route` or `"All Routes"`. Distractor `N004` (routes outside dataset) and cross-route citations are blocked.
2. **Temporal Window & Route-Specific Horizon**:
   - `Chennai-Bangalore` flooding (`N001`, Feb 24 to Mar 8, passable Mar 9) justifies `2025-02-24` and `2025-03-03`, but is overridden to `Yes` (Unexplained) on `2025-03-10` and `2025-03-17`.
   - Route-specific notes without an explicit end date (such as `N002` on `Ahmedabad-Mumbai` if an enricher returns `effective_end_date = null`) default to a strict **1-week window** (`start_date + 6 days`), preventing single-week surcharges from justifying spikes weeks later.
3. **Causal Direction**: Notes with `cost_effect != "increase"` (`N005` minor delays, `N006` stable demand, `N007` improved roads, `N008` normal movement, `N009` repairs completed, `N010` absorbed compliance costs) cannot justify a cost rise.
4. **Nationwide (`All Routes`) Peer Divergence Gate**: A nationwide cost shift (`N003`, diesel +5-7%) affects peer corridors equally. If a route diverges from same-week peers by more than `ALL_ROUTES_MAX_PEER_GAP_PCT` (10.0%), such as `Mumbai-Pune` on `2025-06-23` (+6.2% own, +20.6% peer), `N003` is rejected.
5. **Magnitude Ceiling (`Mumbai-Delhi 2025-05-12` Decision)**: `N003` states a 5-7% nationwide diesel impact. With a +3.0% tolerance ceiling (10.0% maximum), `Mumbai-Delhi` on `2025-05-05` (+7.3% own) is not flagged as an acute spike (<10%), whereas `Mumbai-Delhi` on `2025-05-12` (+11.6% own, +6.7% peer) exceeds the 10.0% ceiling and is deliberately flagged as `Yes` (Unexplained) for human review.

---

## 4. How I Know It Is Right (Verification & Evaluation Harness)

1. **Dual-Engine Baseline Cross-Check (`tests/test_baselines.py`)**: Cross-validates the streaming `deque` engine against an independent vectorized `pandas` reference across all 728 rows (`rtol=1e-12`), verifies the 3 sample rows (`Delhi-Jaipur 2024-11-11`, `Ahmedabad-Mumbai 2025-01-20`, `Mumbai-Pune 2025-09-15`), and proves zero look-ahead bias.
2. **End-to-End Labeled Gold Set & Counterfactual Suite (`src/evaluation/eval_harness.py`)**:
   - Scores the pipeline against all 22 labeled candidate route-weeks (`verdict_accuracy = 1.0`, `citation_accuracy = 1.0`) and outputs `outputs/eval_report.json`.
   - Executes end-to-end adversarial counterfactuals through `AnomalyVerifier.verify_candidate()` (forcing an adversarial client to cite `N003` on peer-divergent `Mumbai-Pune 2025-06-23`, `N002` with null end date 7 weeks late, `N001` post-flood on `2025-03-10`, and `N003` on `+11.6%` `Mumbai-Delhi 2025-05-12`), verifying 100% rejection to `Yes`.
3. **3-Process Cold Reproducibility Check (`src/evaluation/repro_check.py`)**: Runs `python -m src.main --no-cache` in 3 separate subprocesses, asserting 100% agreement across all graded columns and SHA-256 hashes (`outputs/reproducibility_report.txt`).

---

## 5. Setup, Execution & Cost Accounting

### 5.1 Setup
Note: On the very first run, ChromaDB downloads the `all-MiniLM-L6-v2` ONNX embedding model (~79 MB) to `~/.cache/chroma/`. In Docker builds, this model is pre-baked into the image during build time.

```bash
python -m venv venv
# Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Add your GROQ_API_KEY or GEMINI_API_KEY in .env

### 5.3 Token & Cost Accounting (`outputs/token_cost_log.json`)

By pruning 706 of 728 route-weeks (97.0%) before LLM verification, a full cold run makes **23 LLM calls** (1 single batch note enrichment + 22 candidate verifications; 22 calls on warm cache)[cite: 1, 2].

- **Provider / Model**: `groq` / `openai/gpt-oss-120b` (published rate: $0.15 / 1M input tokens, $0.60 / 1M output tokens)
- **Total LLM Calls**: 23 calls
- **Total Input Tokens**: 13,785 tokens[cite: 2]
- **Total Output Tokens**: 11,257 tokens[cite: 2]
- **Total Operational Run Cost**: **$0.017026 USD (~1.7 cents)**[cite: 2]
- **Full Audit Log**: Logged per call with timestamps, token counts, and latency in `outputs/token_cost_log.json`[cite: 2].