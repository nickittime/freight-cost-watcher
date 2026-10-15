# src/evaluation/repro_check.py
import hashlib
from pathlib import Path
import subprocess
import sys

import pandas as pd

from src.config import get_settings
from src.logger import get_logger

logger = get_logger(__name__)

GRADED_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
]


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def run_3_pass_reproducibility_check() -> None:
    settings = get_settings()
    settings.ensure_dirs()
    report_path = settings.OUTPUT_DIR / "reproducibility_report.txt"
    csv_path = settings.OUTPUT_DIR / "flagged_output.csv"
    audit_path = settings.OUTPUT_DIR / "llm_audit.csv"

    print("=================================================================")
    print("EXECUTING 3-PROCESS COLD REPRODUCIBILITY CHECK (--no-cache)")
    print("=================================================================\n")

    runs: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    hashes: list[str] = []

    for i in range(1, 4):
        logger.info("Launching subprocess Pass %d/3...", i)
        subprocess.run(
            [sys.executable, "-m", "src.main", "--no-cache"],
            check=True,
        )
        df_sub = pd.read_csv(csv_path, keep_default_na=False)
        df_audit = pd.read_csv(audit_path, keep_default_na=False)
        runs.append(df_sub)
        audits.append(df_audit)

        file_hash = compute_sha256(csv_path)
        hashes.append(file_hash)
        print(f"Pass {i} Complete | Rows: {len(df_sub)} | SHA-256: {file_hash}")

    df1, df2, df3 = runs[0], runs[1], runs[2]
    a1, a2, a3 = audits[0], audits[1], audits[2]

    # 1. Strict assertions on all graded columns across all 3 passes
    for col in GRADED_COLUMNS:
        assert (df1[col] == df2[col]).all(), f"Graded column '{col}' diverged Pass 1 vs 2"
        assert (df1[col] == df3[col]).all(), f"Graded column '{col}' diverged Pass 1 vs 3"

    # 2. Strict assertion on final reason column across all 3 passes
    reasons_identical = bool(
        (df1["reason"] == df2["reason"]).all() and (df1["reason"] == df3["reason"]).all()
    )
    assert reasons_identical, "Final reason column diverged across passes"

    # 3. Strict assertion on full CSV SHA-256 bit-stability
    hashes_identical = len(set(hashes)) == 1
    assert hashes_identical, f"SHA-256 hashes diverged across passes: {hashes}"

    # 4. Telemetry: Audit raw LLM proposal consistency vs guardrail stabilization
    total_candidates = len(a1)
    raw_agree_1_2 = int((a1["raw_llm_verdict"] == a2["raw_llm_verdict"]).sum())
    raw_agree_1_3 = int((a1["raw_llm_verdict"] == a3["raw_llm_verdict"]).sum())
    min_agree = min(raw_agree_1_2, raw_agree_1_3)
    raw_agreement_pct = round(min_agree / total_candidates * 100, 1)

    overrides_p1 = int(a1["gate_overridden"].astype(bool).sum())
    overrides_p2 = int(a2["gate_overridden"].astype(bool).sum())
    overrides_p3 = int(a3["gate_overridden"].astype(bool).sum())

    divergent_details = []
    for idx in range(total_candidates):
        v1 = a1.iloc[idx]["raw_llm_verdict"]
        v2 = a2.iloc[idx]["raw_llm_verdict"]
        v3 = a3.iloc[idx]["raw_llm_verdict"]
        if not (v1 == v2 == v3):
            r_name = a1.iloc[idx]["route"]
            w_date = a1.iloc[idx]["week_of"]
            final_v = df1.iloc[idx]["flagged"]
            divergent_details.append(
                f"  - {r_name} ({w_date}): Raw LLM proposed [{v1}, {v2}, {v3}] -> "
                f"Guarded verdict resolved deterministically to '{final_v}'"
            )

    report_lines = [
        "FREIGHT COST WATCHER - 3-RUN REPRODUCIBILITY VERIFICATION REPORT",
        "=================================================================",
        f"Provider / Model : {settings.LLM_PROVIDER} / {settings.GROQ_MODEL}",
        f"Pass 1 SHA-256   : {hashes[0]}",
        f"Pass 2 SHA-256   : {hashes[1]}",
        f"Pass 3 SHA-256   : {hashes[2]}",
        "",
        "FINAL SUBMISSION CONTRACT VERIFICATION:",
        "- Graded fields (route, week_of, cost_per_tonne_km, vs_own_history, vs_similar_routes, flagged, matched_note_id): 100% IDENTICAL",
        "- Reason column: 100% IDENTICAL across all 3 passes",
        "- Full CSV SHA-256 hashes: 100% IDENTICAL across all 3 passes",
        f"- Total candidate anomalies in output: {len(df1)}",
        "",
        "LLM REASONING & GUARDRAIL AUDIT TELEMETRY (outputs/llm_audit.csv):",
        f"- Raw LLM proposal consistency: {raw_agreement_pct}% ({min_agree}/{total_candidates} candidates)",
        f"- Deterministic gate overrides: Pass 1: {overrides_p1}, Pass 2: {overrides_p2}, Pass 3: {overrides_p3}",
    ]

    if divergent_details:
        report_lines.append("- Stochastic LLM divergences neutralized by deterministic guardrails:")
        report_lines.extend(divergent_details)
    else:
        report_lines.append("- Raw LLM proposals were 100% identical across all runs.")

    report_lines.append("=================================================================")

    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    run_3_pass_reproducibility_check()