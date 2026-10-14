# Freight Cost Watcher

A production-grade, reproducible AI decision system that monitors weekly road freight costs across shipping corridors, detects acute rate spikes and sustained cost creep, and verifies candidate anomalies against operational context notes using hybrid RAG and deterministic guardrails.

---

## 1. System Topology & Architecture

```
                               +-----------------------------+
                               |    shipment_records.csv     |
                               |    (2,940 raw shipments)    |
                               +--------------+--------------+
                                              |
                                              v
                              +-------------------------------+
                              |    Stage 1: Validation &      |
                              |   Ratio-of-Sums Aggregation   |
                              |  (7 routes x 104 Mondays =    |
                              |       728 route-weeks)        |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              |   Stage 2: Baseline Engine    |
                              | - Own History: O(8) deque     |
                              |   (calendar 56-day rolling)   |
                              | - Peer LOO: O(1) route_type   |
                              +---------------+---------------+
                                              |
                                              v
                              +-------------------------------+
                              | Stage 3: Anomaly Gate         |
                              | - Acute Spike (>=10%)         |
                              | - Hysteresis Creep (20%->15%) |
                              | -> 22 Candidates (97% pruned) |
                              +---------------+---------------+
                                              | (Candidate Pool)
+-----------------------+                     v
|   context_notes.csv   |        +----------------------------+
|  (10 real-world notes)|        | Stage 4: Grounded Verifier |
+-----------+-----------+        | - ChromaDB RAG (Top-5)     |
            |                    | - Groq gpt-oss-120b (T=0)  |
            v                    +--------------+-------------+
+-----------------------+                       |
| One-Time Note Indexer |                       v
| & Structured Metadata |        +----------------------------+
| (Scope, Window, Mag)  |------->| Stage 5: Deterministic     |
+-----------------------+        | Guardrail Gate             |
                                 | (Route, Window, Magnitude, |
                                 |  Nationwide Peer Gap)      |
                                 +--------------+-------------+
                                                |
               +--------------------------------+--------------------------------+
               |                                |                                |
               v                                v                                v
  +-------------------------+      +-------------------------+      +-------------------------+
  | outputs/                |      | outputs/                |      | Stretch Goal:           |
  | flagged_output.csv      |      | token_cost_log.json     |      | Interactive CLI Agent   |
  | (Submission Contract)   |      | (Tokens & Published USD)|      | (Dynamic Tool Use)      |
  +-------------------------+      +-------------------------+      +-------------------------+
```

---

## 2. Mathematical Definitions & Baseline Modeling

### 2.1 Group-Level Ratio-of-Sums
Weekly cost per tonne-km is calculated strictly at the group level as the ratio of sums (freight cost divided by payload-distance), not as the average of shipment ratios:
$$\text{cost\_per\_tonne\_km}_{r, w} = \frac{\sum_{i \in (r, w)} \text{freight\_cost\_inr}_i}{\sum_{i \in (r, w)} (\text{quantity\_tonnes}_i \times \text{distance\_km}_i)}$$
Averaging individual shipment ratios would distort corridor economics by giving disproportionate weight to lightweight shipments.

### 2.2 Own-History Trailing Window (Zero Look-Ahead)
- Enforces an $O(\text{window}) = O(8)$ calendar-bounded rolling deque per route over strictly prior Mondays (maximum lookback: 56 calendar days).
- Evaluates strictly prior weeks (`shift(1)`), preventing data leakage.
- **Missing History Policy**: For routes with fewer than 8 prior weeks, the baseline averages all available prior weeks without artificial padding or extrapolation. A route's first week has no prior baseline and is formatted as `"N/A (no prior history)"`.

### 2.3 Peer Average Baseline (Leave-One-Out)
For each week and `route_type` bucket (Short, Medium, Long), the peer baseline excludes the evaluated route itself:
$$\text{peer\_baseline}_{r, w} = \frac{\sum_{j \in \text{type}(r)} \text{cptk}_{j, w} - \text{cptk}_{r, w}}{N_{\text{type}(r)} - 1}$$
Computed in constant $O(1)$ time per row from bucket aggregates.

---

## 3. Anomaly Detection & Hysteresis Design

Standard thresholding flags acute price jumps but fails on quiet cost creep. On `Mumbai-Pune`, costs rose gradually from ~₹3.41 to ₹4.49 in late 2025. Because the rise was continuous, the trailing 8-week baseline adjusted upward with the route, keeping `vs_own_history` modest (+5% to +9%) while `vs_similar_routes` surged to +38.9%.

A static threshold of `vs_peer >= 20% AND vs_own >= 5%` toggles erratically across consecutive weeks. We implemented a **hysteresis state machine**:
1. **Acute Spike Trigger**: `vs_own_pct >= 10.0%`.
2. **Sustained Creep Entry**: `vs_peer_pct >= 20.0%` AND `vs_own_pct >= 5.0%`.
3. **Hysteresis Sustain**: Once entered, the route remains flagged while `vs_peer_pct >= 15.0%` AND `vs_own_pct > 0.0%`.

### Threshold Sensitivity Analysis
Generated directly from `AnomalyDetector.run_sensitivity_analysis()` across all 728 route-weeks:

| Spike Threshold | Creep Peer Entry | Min Own Rise | Hysteresis Keep | Candidate Pool | Behavior |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **10.0%** | **20.0%** | **5.0%** | **15.0%** | **22 rows** | **Baseline Configuration** |
| 8.0% | 20.0% | 5.0% | 15.0% | 23 rows | +1 marginal acute week |
| 12.0% | 20.0% | 5.0% | 15.0% | 21 rows | -1 marginal acute week |
| 10.0% | 18.0% | 4.0% | 12.0% | 22 rows | Identical candidate set |
| 10.0% | 22.0% | 6.0% | 15.0% | 19 rows | Stricter creep entry (-3 weeks) |

---

## 4. Anti-Hallucination Guardrails & Edge Cases

Every `"justified"` verdict proposed by the LLM must pass deterministic verification rules in `src/guardrails.py` before inclusion:

1. **Route Scope Gate**: `note.applies_to` must match `route` or `"All Routes"`. Distractor note `N004` (affecting corridors outside the dataset) is blocked.
2. **Temporal Window & Route-Specific Horizon**:
   - `Chennai-Bangalore` flooding (`N001`, Feb 24 to Mar 8, passable Mar 9) justifies weeks `2025-02-24` and `2025-03-03`, but is overridden to `Yes` (Unexplained) on `2025-03-10` and `2025-03-17`.
   - Route-specific notes with open-ended or null end dates (e.g., `N002` on `Ahmedabad-Mumbai`) default to a strict **1-week window** (`start_date + 6 days`), preventing short-term surcharges from justifying cost increases weeks later.
3. **Causal Direction Gate**: Notes describing absorbed costs (`N010`), unaffected costs (`N005`), or normal operations (`N006`, `N007`, `N008`, `N009`) cannot justify a cost increase.
4. **Nationwide (`All Routes`) Peer Divergence Gate**: A nationwide cost shift (`N003`, diesel +5–7%) affects peer routes equally. If a corridor diverges from same-week peers by more than 10.0% (e.g., `Mumbai-Pune` on `2025-06-23` with +6.2% own, +20.6% peer), `N003` is rejected because general fuel increases cannot explain corridor-specific divergence.
5. **Magnitude Ceiling (`Mumbai-Delhi 2025-05-12`)**: `N003` states diesel rose by roughly 5–7%. With a +3.0% tolerance ceiling (10.0% max allowed), `Mumbai-Delhi` on `2025-05-12` (+11.6% own, +6.7% peer) exceeds this boundary and is deliberately flagged as `Yes` (Unexplained) for human audit.

---

## 5. Verification, Evaluation Harness & Reproducibility

### 5.1 Dual-Engine Baseline Cross-Check
The streaming `deque` baseline engine is cross-checked against a vectorized `pandas` rolling reference across all 728 route-weeks (`rtol=1e-12`), ensuring zero numerical drift and confirming zero look-ahead data leakage.

### 5.2 End-to-End Evaluation Harness (`src/evaluation/eval_harness.py`)
- Evaluates the pipeline against an independently authored **22-row Gold Set** established from raw context notes and ground-truth logistics events:
  - **Verdict Accuracy**: 100% (22/22)
  - **Citation Accuracy**: 100% (22/22)
- Executes 4 end-to-end adversarial counterfactual tests forcing an adversarial client to propose invalid citations, confirming that the deterministic guardrail intercepts all 4 traps and forces verdicts to `Yes` (Unexplained).

### 5.3 3-Process Cold Reproducibility Proof
Because `openai/gpt-oss-120b` is an open-weights reasoning model that expends reasoning tokens per completion (~490 output tokens per call), reason text is synthesized deterministically from verified structured fields. Across 3 separate cold subprocesses (`python -m src.main --no-cache`), all graded columns and CSV SHA-256 hashes are 100% bit-identical.

---

## 6. Setup, Execution & Cost Accounting

### 6.1 Installation
On first run, ChromaDB downloads the `all-MiniLM-L6-v2` ONNX model (~79 MB) to `~/.cache/chroma/`. In Docker builds, this model is baked into the image.

```bash
python -m venv venv
# Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env
# Set GROQ_API_KEY in .env
```

### 6.2 Execution Commands
```bash
# 1. Full pipeline run (generates flagged_output.csv & token_cost_log.json)
python -m src.main --no-cache

# 2. End-to-end evaluation harness (generates eval_report.json)
python -m src.evaluation.eval_harness

# 3. 3-process cold reproducibility check (generates reproducibility_report.txt)
python -m src.evaluation.repro_check

# 4. Run automated pytest test suite
pytest

# 5. Interactive tool-calling agent CLI
python -m src.main --ask "Why did Chennai-Bangalore get pricier in February and March 2025?"
```

### 6.3 Operational Token & Cost Accounting (`outputs/token_cost_log.json`)
By pruning 706 of 728 route-weeks (97.0%) prior to LLM verification, a cold run requires **23 LLM calls** (1 batch note enrichment + 22 candidate verifications; 22 calls on warm cache).

- **Provider / Model**: `groq` / `openai/gpt-oss-120b` (temp=0.0, seed=42)
- **Published Rates**: $0.15 / 1M input tokens, $0.60 / 1M output tokens (checked October 2026)
- **Total LLM Calls**: 23 calls
- **Input Tokens**: 13,785 tokens ($0.002068 USD)
- **Output Tokens**: 11,257 tokens ($0.006754 USD)
- **Total Operational Cost**: **$0.008822 USD (under 1 cent)**

---

## 7. Assumptions, Design Tradeoffs & Future Work

### Assumptions
1. **Directional Corridors**: Routes are treated as directed pairs (`Origin-Destination`). `Mumbai-Pune` and `Pune-Mumbai` represent distinct freight corridors with different elevation profiles, toll structures, and backhaul demand.
2. **Monday-Anchored Weeks**: Weeks are identified by the Monday date (`YYYY-MM-DD`) of the shipment week.
3. **Conservative Review Policy**: Any candidate anomaly lacking verified, unambiguous evidence degrades to `Yes` (Unexplained). In logistics operations, flagging an anomaly for human audit carries significantly lower business risk than providing a false justification.

### Design Tradeoffs
1. **Deterministic Reason Synthesis vs. Free-Form Generation**: Rather than passing raw LLM text into the final CSV, reason explanations are synthesized deterministically from verified structured fields. This trades stylistic variety for guaranteed non-contradiction and bit-stable CSV reproducibility across runs.
2. **Hybrid Scope-Complete Retrieval vs. Pure Dense Search**: Dense semantic search alone can miss short temporal notes. Our hybrid retriever guarantees the inclusion of all notes scoped to the corridor or `"All Routes"`, ranked by date proximity and dense similarity.

### Future Work
1. **Dynamic Seasonal Indexing**: Incorporating multi-year calendar awareness (e.g., Diwali and regional festival shifts) into baseline calculations.
2. **Carrier-Level Fixed-Effects Decomposition**: Decomposing freight cost increases into transporter-specific rate changes versus general corridor cost inflation.
3. **Asynchronous Batched Verification**: Utilizing asynchronous client pools for processing enterprise shipment datasets covering thousands of active corridors.