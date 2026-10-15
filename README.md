# Freight Cost Watcher

A production-grade, reproducible AI decision system that monitors weekly road freight costs across shipping corridors, detects acute rate spikes and sustained cost creep, and verifies candidate anomalies against operational context notes using hybrid RAG and deterministic guardrails.

---

## 1. System Architecture

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
|                    | - Groq gpt-oss-20b (T=0)   |
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


---

## 2. Mathematical Modeling & Baseline Engine

### 2.1 Group-Level Ratio-of-Sums
Corridor freight economics cannot be calculated as the average of individual shipment cost ratios, as that gives disproportionate mathematical weight to lightweight shipments. Weekly unit cost per tonne-km is calculated strictly at the group level as the ratio of sums:

$$\text{cost\_per\_tonne\_km}_{r, w} = \frac{\sum_{i \in (r, w)} \text{freight\_cost\_inr}_i}{\sum_{i \in (r, w)} (\text{quantity\_tonnes}_i \times \text{distance\_km}_i)}$$

### 2.2 Own-History Trailing Baseline (Zero Look-Ahead)
- Enforces an $O(\text{window}) = O(8)$ calendar-bounded rolling deque per corridor over strictly prior Mondays (maximum 56 calendar days).
- Evaluates strictly prior weeks (`shift(1)`), preventing look-ahead data leakage.
- **Missing History Policy**: For routes with fewer than 8 prior weeks, the baseline averages available prior weeks without artificial padding or extrapolation. The initial week of a corridor has no prior history and is formatted as `"N/A (no prior history)"`.

### 2.3 Peer Corridor Baseline (Leave-One-Out)
For each week and corridor distance category (`Short`, `Medium`, `Long`), the peer baseline excludes the corridor itself to avoid self-influence:

$$\text{peer\_baseline}_{r, w} = \frac{\sum_{j \in \text{type}(r)} \text{cptk}_{j, w} - \text{cptk}_{r, w}}{N_{\text{type}(r)} - 1}$$

Computed in constant $O(1)$ time per row from bucket aggregates.

---

## 3. Anomaly Detection & Hysteresis Design

Static thresholding detects acute price spikes but fails on quiet cost creep. On `Mumbai-Pune` during late 2025, freight rates climbed gradually from ~₹3.41 to ₹4.49 per tonne-km. Because the increase was continuous, the trailing 8-week baseline adjusted upward with the corridor, keeping own-history deviation modest (+5% to +9%) while peer divergence climbed to +38.9%.

Static dual thresholds toggle erratically across consecutive weeks. We implemented a **hysteresis state machine**:
1. **Acute Spike Trigger**: `vs_own_pct >= 10.0%`.
2. **Sustained Creep Entry**: `vs_peer_pct >= 20.0%` AND `vs_own_pct >= 5.0%`.
3. **Hysteresis Sustain**: Once entered, the corridor remains flagged as long as `vs_peer_pct >= 15.0%` AND `vs_own_pct > 0.0%`.

### Threshold Sensitivity Analysis
Generated across all 728 route-weeks:

| Spike Threshold | Creep Peer Entry | Min Own Rise | Hysteresis Keep | Candidate Pool | Behavior |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **10.0%** | **20.0%** | **5.0%** | **15.0%** | **22 rows** | **Baseline Configuration** |
| 8.0% | 20.0% | 5.0% | 15.0% | 23 rows | +1 marginal acute week |
| 12.0% | 20.0% | 5.0% | 15.0% | 21 rows | -1 marginal acute week |
| 10.0% | 18.0% | 4.0% | 12.0% | 22 rows | Identical candidate set |
| 10.0% | 22.0% | 6.0% | 15.0% | 19 rows | Stricter creep entry (-3 weeks) |

---

## 4. Deterministic Anti-Hallucination Guardrails

Every justification proposed by the LLM must pass pure Python deterministic validation checks before inclusion:

1. **Corridor Scope Gate**: `note.applies_to` must match `route` or `"All Routes"`. Distractor note `N004` (affecting corridors outside the dataset) is blocked.
2. **Temporal Window Gate**:
   - `Chennai-Bangalore` flooding (`N001`, Feb 24 to Mar 8, passable Mar 9) justifies weeks `2025-02-24` and `2025-03-03`. Lingering weeks `2025-03-10` and `2025-03-17` fall outside the flood window and are forced to `Yes` (Unexplained).
   - Route-specific notes with null end dates (e.g., `N002` on `Ahmedabad-Mumbai`) default to a **1-week window** (`start_date + 6 days`), preventing short-term surcharges from justifying increases months later.
3. **Causal Direction Gate**: Notes describing absorbed costs (`N010`), routine maintenance (`N005`), or normal operations (`N006`, `N007`, `N008`, `N009`) cannot justify an increase.
4. **Nationwide Peer Gap Gate**: A nationwide shift (`N003`, diesel +5–7%) affects peer routes equally. If a corridor diverges from same-week peers by more than 10.0% (e.g., `Mumbai-Pune` on `2025-06-23` with +6.2% own, +20.6% peer), `N003` is rejected.
5. **Magnitude Ceiling Gate**: `Mumbai-Delhi` on `2025-05-12` saw an 11.6% rise, exceeding the stated diesel impact of 5–7% (+3.0% tolerance ceiling = 10.0% max). It is flagged as `Yes` (Unexplained) for human review.

---

## 5. Evaluation, Planted Benchmarks & Reproducibility

### 5.1 Detector Recall on Planted Benchmark Events
To verify the detector does not suffer from false negatives, recall was evaluated against 9 planted operational events:
- **Operational Notes**: `N001` flooding weeks (`2025-02-24`, `2025-03-03`) and `N002` festival surcharge (`2025-01-20`).
- **Empirical Corridors**: Acute rate surges on `Delhi-Jaipur` (`2024-11-11`, `2024-11-18`), post-flood lingering weeks on `Chennai-Bangalore` (`2025-03-10`), diesel ceiling excess on `Mumbai-Delhi` (`2025-05-12`), and creep entry/peak on `Mumbai-Pune` (`2025-09-15`, `2025-12-22`).
- **Result**: **100% Detector Recall** (9/9 benchmark events captured in candidate pool).

### 5.2 Verifier Accuracy & Adversarial Traps (`outputs/eval_report.json`)
- Evaluated against candidate pool ground truth:
  - **Candidate Verifier Accuracy**: 100% (22/22)
  - **Citation Precision**: 100% (22/22)
- Evaluated against 4 adversarial counterfactual scenarios (forcing an adversarial client to propose invalid notes). The guardrail gate intercepted all 4 traps.

### 5.3 3-Process Cold Reproducibility Proof (`outputs/reproducibility_report.txt`)
Across 3 independent cold subprocesses (`python -m src.main --no-cache`), all graded columns and CSV SHA-256 hashes are 100% bit-identical (`dfd05c28dc1fe65c3d122f2c347b3acc60b8f7b6b3b190292c9733662ff42023`).

---

## 6. Setup, Execution & Cost Accounting

### 6.1 Setup
```bash
python -m venv venv
# Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env
# Set GROQ_API_KEY in .env
6.2 Execution Commands
Bash
# 1. Run unit test suite (39 tests, offline)
python -m pytest

# 2. Run offline evaluation harness (0 tokens)
python -m src.evaluation.eval_harness

# 3. Run 3-process cold reproducibility harness
python -m src.evaluation.repro_check

# 4. Run main pipeline
python -m src.main --no-cache

# 5. Interactive tool-calling agent CLI (Stretch Goal)
python -m src.main --ask "Why did Chennai-Bangalore get pricier in February and March 2025?"
6.3 Operational Token & Cost Accounting (outputs/token_cost_log.json)
Pre-gating prunes 706 of 728 route-weeks (97.0%) prior to LLM verification.

Provider / Model: groq / openai/gpt-oss-20b (temp=0.0, seed=42)

Published Rates: $0.075 / 1M input tokens, $0.30 / 1M output tokens (checked October 2026)

Total LLM Calls: 19 calls (19 unique requests, 0 retries)

Input Tokens: 11,381 tokens ($0.000854 USD)

Output Tokens: 9,230 tokens ($0.002769 USD)

Total Tokens: 20,611

Total Operational Cost: $0.003623 USD (approx. 0.36 cents)

7. Assumptions, Tradeoffs & Future Work
Assumptions
Directional Corridors: Corridors are treated as directed pairs (Origin-Destination) because elevation profiles, toll structures, and backhaul demand differ by direction.

Monday-Anchored Weeks: Weeks are identified by the Monday date (YYYY-MM-DD) of the shipment week.

Conservative Review Policy: Candidate anomalies lacking unambiguous evidence degrade to Yes (Unexplained). Flagging an anomaly for human audit carries lower operational risk than a false justification.

Design Tradeoffs
Diagnostic Reason Synthesis vs. Free-Form Generation: Rather than passing raw LLM completion text into the final CSV, reason explanations are synthesized from verified diagnostic checks. This trades stylistic variation for guaranteed non-contradiction and bit-stable CSV reproducibility across runs.

Hybrid Scope-Complete Retrieval vs. Pure Dense Search: Dense semantic search alone can miss short temporal notes. Our hybrid retriever guarantees the inclusion of all notes scoped to the corridor or "All Routes", ranked by temporal proximity.

Future Work
Dynamic Seasonal Indexing: Incorporating multi-year calendar awareness (e.g., regional festival shifts) into baseline calculations.

Carrier Fixed-Effects Decomposition: Decomposing freight cost increases into transporter-specific rate changes versus corridor-wide inflation.

Asynchronous Batched Verification: Utilizing asynchronous client pools for processing enterprise shipment datasets covering thousands of active corridors.


---

### Step 2: Push to GitHub

Once saved in Notepad, stage, commit, and push:

```cmd
git add README.md
git commit -m "docs: finalize README with escaped LaTeX underscores and formatted markdown tables"
git push origin main