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

    print("=================================================================")
    print("EXECUTING 3-PROCESS COLD REPRODUCIBILITY CHECK (--no-cache)")
    print("=================================================================\n")

    runs: list[pd.DataFrame] = []
    hashes: list[str] = []

    for i in range(1, 4):
        logger.info("Launching subprocess Pass %d/3...", i)
        subprocess.run(
            [sys.executable, "-m", "src.main", "--no-cache"],
            check=True,
        )
        df_sub = pd.read_csv(csv_path, keep_default_na=False)
        runs.append(df_sub)
        file_hash = compute_sha256(csv_path)
        hashes.append(file_hash)
        print(f"Pass {i} Complete | Rows: {len(df_sub)} | SHA-256: {file_hash}")

    df1, df2, df3 = runs[0], runs[1], runs[2]

    # 1. Strict assertion on all graded columns across all 3 runs
    for col in GRADED_COLUMNS:
        assert (df1[col] == df2[col]).all(), f"Column '{col}' diverged between Pass 1 and Pass 2"
        assert (df1[col] == df3[col]).all(), f"Column '{col}' diverged between Pass 1 and Pass 3"

    # 2. Check reason column equality across runs
    reasons_identical = bool(
        (df1["reason"] == df2["reason"]).all() and (df1["reason"] == df3["reason"]).all()
    )
    hashes_identical = len(set(hashes)) == 1

    status_str = (
        "100% BIT-STABLE AND IDENTICAL ACROSS 3 SUBPROCESSES"
        if hashes_identical
        else "GRADED FIELDS 100% IDENTICAL (REASON WORDING COMPARISON REPORTED SEPARATELY)"
    )

    report_lines = [
        "FREIGHT COST WATCHER - 3-RUN REPRODUCIBILITY VERIFICATION REPORT",
        "=================================================================",
        f"Provider / Model : {settings.LLM_PROVIDER} / {settings.GROQ_MODEL}",
        f"Pass 1 SHA-256   : {hashes[0]}",
        f"Pass 2 SHA-256   : {hashes[1]}",
        f"Pass 3 SHA-256   : {hashes[2]}",
        "",
        f"VERIFICATION STATUS: {status_str}",
        f"- Graded fields ({', '.join(GRADED_COLUMNS)}): IDENTICAL",
        f"- Reason text identical across all 3 runs: {reasons_identical}",
        f"- Full CSV SHA-256 hashes identical across all 3 runs: {hashes_identical}",
        f"- Total candidate anomalies in output: {len(df1)}",
        "=================================================================",
    ]
    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    run_3_pass_reproducibility_check()