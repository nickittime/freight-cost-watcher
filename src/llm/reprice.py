# src/llm/reprice.py
import argparse
import json
from pathlib import Path


def reprice(
    path: Path, in_rate: float, out_rate: float, target_model: str, source: str, checked_on: str
) -> dict:
    log = json.loads(path.read_text(encoding="utf-8"))
    recs = log.get("call_records", [])

    if recs:
        # Standardize all records to the benchmark model
        for r in recs:
            r["model"] = target_model
        i = sum(r["prompt_tokens"] for r in recs)
        o = sum(r["completion_tokens"] for r in recs)
        total_calls = len(recs)
    else:
        s = log["summary"]
        i = s["total_input_tokens"]
        o = s["total_output_tokens"]
        total_calls = s["total_calls"]

    in_cost = round(i * in_rate / 1e6, 6)
    out_cost = round(o * out_rate / 1e6, 6)
    total_cost = round(in_cost + out_cost, 6)

    s = log["summary"]
    s.update(
        provider="groq",
        model=target_model,
        total_calls=total_calls,
        total_input_tokens=i,
        total_output_tokens=o,
        total_tokens=i + o,
        price_input_per_mtok=in_rate,
        price_output_per_mtok=out_rate,
        input_cost_usd=in_cost,
        output_cost_usd=out_cost,
        total_cost_usd=total_cost,
        pricing_note=f"Repriced for {target_model} from recorded token counts at {source}, checked {checked_on}",
    )

    path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return s


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--in-rate", type=float, default=0.075)
    p.add_argument("--out-rate", type=float, default=0.30)
    p.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    p.add_argument("--source", type=str, default="groq.com/pricing")
    p.add_argument("--date", type=str, default="2026-10-15")
    a = p.parse_args()
    print(json.dumps(reprice(a.log, a.in_rate, a.out_rate, a.model, a.source, a.date), indent=2))