\# Freight Cost Watcher — Production AI Assistant



A production-grade, reproducible AI decision system that monitors road freight costs, isolates unjustified cost creep, and verifies shipping rate anomalies against external disruption notes using deterministic guardrails and hybrid RAG.



\---



\## 1. System Topology \& Architecture





&#x20;                          +-----------------------------+

&#x20;                          |    shipment\_records.csv     |

&#x20;                          |    (2,940 raw shipments)    |

&#x20;                          +--------------+--------------+

&#x20;                                         |

&#x20;                                         v

&#x20;                         +-------------------------------+

&#x20;                         |    Stage 1: Validation \&      |

&#x20;                         |   Ratio-of-Sums Aggregation   |

&#x20;                         |  (7 routes x 104 Mondays =    |

&#x20;                         |       728 route-weeks)        |

&#x20;                         +---------------+---------------+

&#x20;                                         |

&#x20;                                         v

&#x20;                         +-------------------------------+

&#x20;                         |   Stage 2: Baseline Engine    |

&#x20;                         | - Own History: O(8) deque     |

&#x20;                         |   (calendar 56-day rolling)   |

&#x20;                         | - Peer LOO: O(1) route\_type   |

&#x20;                         +---------------+---------------+

&#x20;                                         |

&#x20;                                         v

&#x20;                         +-------------------------------+

&#x20;                         | Stage 3: Anomaly Gate         |

&#x20;                         | - Acute Spike (>=10%)         |

&#x20;                         | - Hysteresis Creep (20%->15%) |

&#x20;                         | -> 22 Candidates (97% pruned) |

&#x20;                         +---------------+---------------+

&#x20;                                         |(Candidate Pool)

&#x20;                                         -------------------+

&#x09;						     |

&#x20;                                                            v

&#x09;	+-----------------------+        +----------------------------+             

&#x09;	|   context\_notes.csv   |        | Stage 4: Grounded Verifier |

&#x09;	|  (10 real-world notes)|        | - ChromaDB RAG (Top-3)     |

&#x09;	+-----------+-----------+        | - LLM Reasoning (Temp=0.0) |                    

&#x20;          		    v			 +--------------+-------------+		

&#x09;	+-----------------------+                       |

&#x09;	| One-Time Note Indexer |                       v

&#x09;	| \& Structured Metadata |        +----------------------------+

&#x09;	| (Scope, Window, Mag)  |------->| Stage 5: Deterministic     |

&#x09;	+-----------------------+        | Guardrail Gate             |

&#x09;					 | (Route, Window, Magnitude) |

&#x09;					 +--------------+-------------+

&#x09;						        |

&#x09;		       +--------------------------------+--------------------------------+

&#x09;		       |                                |                                |

&#x09;		       v                                v                                v

&#x20;                              +-------------------------+      +-------------------------+      +-------------------------+

&#x20;                              | outputs/                |      | outputs/                |      | Stretch Goal:           |

&#x09;		       | flagged\_output.csv      |      | token\_cost\_log.json     |      | Interactive CLI Agent   |

&#x09;		       | (Submission Contract)   |      | (Tokens \& Published USD)|      | (Multi-step tool use)   |

&#x20;                              +-------------------------+      +-------------------------+      +-------------------------+







\## 2. Core Mathematical Specifications



\### 2.1 Group-Level Ratio-of-Sums

Weekly cost per tonne-km is calculated strictly at the group level as the ratio of sums (freight cost divided by payload-distance), not as the average of shipment ratios:

$$\\text{cost\\\_per\\\_tonne\\\_km}\_{r, w} = \\frac{\\sum\_{i \\in (r, w)} \\text{freight\\\_cost\\\_inr}\_i}{\\sum\_{i \\in (r, w)} (\\text{quantity\\\_tonnes}\_i \\times \\text{distance\\\_km}\_i)}$$



\### 2.2 Own-History Trailing Window (Zero Look-Ahead)

\* Strictly uses weeks prior to the evaluated week (`shift(1)`), preventing look-ahead data leakage.

\* Evaluated over an $O(\\text{window}) = O(8)$ per-route `collections.deque` keyed by calendar days (maximum age: 56 calendar days). 

\* \*\*Missing History Policy\*\*: For routes with fewer than 8 prior weeks, the baseline averages all available prior weeks without artificial padding or extrapolation. The first observed week for any route has no prior baseline and is formatted as `"N/A (no prior history)"`.



\### 2.3 Peer Average Baseline (Leave-One-Out)

For each week and `route\_type` bucket (Short, Medium, Long), the peer baseline excludes the evaluated route itself\[cite: 1]:

$$\\text{peer\\\_baseline}\_{r, w} = \\frac{\\sum\_{j \\in \\text{type}(r)} \\text{cptk}\_{j, w} - \\text{cptk}\_{r, w}}{N\_{\\text{type}(r)} - 1}$$

Computed in $O(1)$ time per row from bucket aggregates.



\---



\## 3. Anomaly Detection \& Hysteresis Design



Standard thresholding flags acute price jumps but fails on quiet cost creep\[cite: 1]. On `Mumbai-Pune`, costs rose gradually from \~₹3.41 to ₹4.49 in late 2025\[cite: 1]. Because the rise was continuous, the trailing 8-week baseline adjusted upward with the route, keeping `vs\_own\_history` modest (+5% to +9%) while `vs\_similar\_routes` surged to +38.9%\[cite: 1].



A static threshold of `vs\_peer >= 20% AND vs\_own >= 5%` toggles erratically, flagging and dropping consecutive weeks arbitrarily\[cite: 1]. We implemented a \*\*hysteresis state machine\*\*:

1\. \*\*Acute Spike Trigger\*\*: `vs\_own\_pct >= 10.0%`\[cite: 1].

2\. \*\*Sustained Creep Entry\*\*: `vs\_peer\_pct >= 20.0%` AND `vs\_own\_pct >= 5.0%`\[cite: 1].

3\. \*\*Hysteresis Sustain\*\*: Once entered, the route remains flagged while `vs\_peer\_pct >= 15.0%` AND `vs\_own\_pct > 0.0%`\[cite: 1].



\### Threshold Sensitivity Grid

Across parameter variations, candidate volume remains stable:



| Spike Threshold | Creep Peer Entry | Min Own Rise | Hysteresis Keep | Candidate Pool | Stability |

| :---: | :---: | :---: | :---: | :---: | :---: |

| \*\*10.0%\*\* | \*\*20.0%\*\* | \*\*5.0%\*\* | \*\*15.0%\*\* | \*\*22 rows\*\* | \*\*Optimal Baseline\*\* |

| 8.0% | 20.0% | 5.0% | 15.0% | 23 rows | +1 marginal row |

| 12.0% | 20.0% | 5.0% | 15.0% | 21 rows | -1 marginal row |

| 10.0% | 18.0% | 4.0% | 12.0% | 22 rows | Completely stable |

| 10.0% | 22.0% | 6.0% | 15.0% | 20 rows | Tight bounds |



\---



\## 4. Anti-Hallucination Guardrail Architecture



LLMs are prone to hallucinating connections when context notes contain related keywords\[cite: 1]. To guarantee that justified verdicts cite only valid evidence\[cite: 1]:



1\. \*\*One-Time Note Enrichment\*\*: At index time, context notes are parsed into immutable metadata: `effective\_start\_date`, `effective\_end\_date`, `cost\_effect` (`increase` | `decrease` | `none`), and `stated\_magnitude\_max\_pct`\[cite: 1].

2\. \*\*Deterministic Gate\*\*: The verifier routes candidate rows through a Python guardrail before finalizing verdicts\[cite: 1]:

&#x20;  \* \*\*Route Scope\*\*: Note `applies\_to` must match the route or `"All Routes"`\[cite: 1].

&#x20;  \* \*\*Temporal Window \& Trap Rejection\*\*: On `Chennai-Bangalore`, Note `N001` explicitly covers flooding between \*\*Feb 24 and Mar 8\*\*, with roads reopening \*\*Mar 9\*\*\[cite: 1]. The guardrail verifies that weeks `2025-02-24` and `2025-03-03` are marked `No (justified)`, while weeks `2025-03-10` and `2025-03-17` are rejected as out-of-window and downgraded to `Yes` (Unexplained)\[cite: 1].

&#x20;  \* \*\*Causal Effect Filter\*\*: Notes describing absorbed costs (`N010`), unaffected costs (`N005`), or normal conditions (`N006`, `N008`, `N009`) are prevented from justifying cost increases\[cite: 1].

&#x20;  \* \*\*Magnitude Boundaries\*\*: Open-ended price hikes like Note `N003` (diesel +5–7%) have an 8-week horizon (matching baseline absorption) and a ±3% tolerance ceiling\[cite: 1]. Observed increases exceeding 10.0% are rejected from being falsely justified by fuel hikes\[cite: 1].



\---



\## 5. Verification \& Reproducibility Proof



\### 5.1 Verification Against Provided Reference

The pipeline reproduces the sample output format rows down to the decimal\[cite: 1, 2]:



| Route | Week Of | cptk | vs\_own\_history | vs\_similar\_routes | Flagged | Matched Note |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: |

| \*\*Delhi-Jaipur\*\*\[cite: 1] | 2024-11-11\[cite: 1] | 4.17\[cite: 1] | +35.5% vs this route's past average\[cite: 1] | +21.0% vs similar-length routes this week\[cite: 1] | Yes\[cite: 1] | \*(blank)\*\[cite: 1] |

| \*\*Ahmedabad-Mumbai\*\*\[cite: 1] | 2025-01-20\[cite: 1] | 3.29\[cite: 1] | +29.5% vs this route's past average\[cite: 1] | +22.5% vs similar-length routes this week\[cite: 1] | No (justified)\[cite: 1] | N002\[cite: 1] |

| \*\*Mumbai-Pune\*\*\[cite: 1] | 2025-09-15\[cite: 1] | 3.98\[cite: 1] | +9.2% vs this route's past average\[cite: 1] | +23.6% vs similar-length routes this week\[cite: 1] | Yes\[cite: 1] | \*(blank)\*\[cite: 1] |



\### 5.2 3-Run Reproducibility

Verifiable via `python -m src.evaluation.repro\_check` across three cold runs without caching:

\* \*\*Pass 1 SHA-256\*\*: `outputs/flagged\_output.csv` (Bit-identical)

\* \*\*Pass 2 SHA-256\*\*: `outputs/flagged\_output.csv` (Bit-identical)

\* \*\*Pass 3 SHA-256\*\*: `outputs/flagged\_output.csv` (Bit-identical)

\* \*\*Result\*\*: Zero numerical divergence, 100% flag consistency\[cite: 2].



\### 5.3 Operational Token \& Inference Cost Accounting

\* \*\*Total Evaluated Route-Weeks\*\*: 728\[cite: 1]

\* \*\*Pre-Gating Candidate Pruning\*\*: 706 rows pruned (96.98% cost reduction)\[cite: 2]

\* \*\*Total LLM Calls\*\*: 22 verification calls + 10 one-time indexing calls = 32 calls\[cite: 2]

\* \*\*Total Tokens\*\*: 10,880 tokens (8,960 input, 1,920 output)\[cite: 2]

\* \*\*Total Run Cost\*\*: \*\*$0.006803 USD\*\* (under 1 cent at published rates of $0.59 / $0.79 per 1M tokens)\[cite: 2]



\---



\## 6. Setup, Execution \& Testing Guide



\### 6.1 Installation

```bash

git clone <repo-url>

cd freight-cost-watcher

python -m venv venv

\# Windows: venv\\Scripts\\activate | Linux/macOS: source venv/bin/activate

pip install -e ".\[dev]"

