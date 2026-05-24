"""Optional LLM-as-Judge evaluator for WayfinderAI TripPlan outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.trip_model import TripPlanResponse  # noqa: E402
from evaluate_trip_plan import extract_trip_payload, load_input, load_trip_from_redis  # noqa: E402

SCORE_KEYS = [
    "preference_satisfaction",
    "practicality",
    "grounding_faithfulness",
    "budget_reasonableness",
    "coherence",
    "overall_quality",
]

SYSTEM_PROMPT = """You are a strict travel-plan evaluator.
Return only JSON. Score every dimension from 1 to 5, where 5 is best.
Evaluate whether the TripPlan is useful, practical, grounded, and coherent.
"""


def build_judge_prompt(plan: dict[str, Any], request: dict[str, Any] | None = None) -> str:
    return f"""Evaluate the following generated travel plan.

Scoring dimensions:
- preference_satisfaction: whether the plan satisfies destination, budget, preferences, pace, and special requirements.
- practicality: whether daily density, route arrangement, hotel placement, and dining choices are realistic.
- grounding_faithfulness: whether attractions, hotels, dining, weather, and coordinates look factual and tool-grounded.
- budget_reasonableness: whether budget fields are internally consistent and reasonable.
- coherence: whether the plan is coherent, specific, and not repetitive.
- overall_quality: overall quality.

Return JSON in this exact shape:
{{
  "scores": {{
    "preference_satisfaction": 1,
    "practicality": 1,
    "grounding_faithfulness": 1,
    "budget_reasonableness": 1,
    "coherence": 1,
    "overall_quality": 1
  }},
  "major_issues": ["issue"],
  "rationale": "short explanation"
}}

User request, if available:
{json.dumps(request or {}, ensure_ascii=False)}

TripPlan:
{json.dumps(plan, ensure_ascii=False)}
"""


def normalize_scores(raw: dict[str, Any]) -> dict[str, float]:
    scores = raw.get("scores") or {}
    normalized: dict[str, float] = {}
    for key in SCORE_KEYS:
        try:
            value = float(scores.get(key, 0))
        except Exception:  # noqa: BLE001
            value = 0.0
        normalized[key] = max(0.0, min(5.0, value))
    return normalized


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def judge_one(plan: dict[str, Any], index: int, dry_run: bool) -> dict[str, Any]:
    TripPlanResponse.model_validate(plan)
    prompt = build_judge_prompt(plan)
    row: dict[str, Any] = {
        "index": index,
        "ok": False,
        "dry_run": dry_run,
        "scores": {key: 0.0 for key in SCORE_KEYS},
        "major_issues": [],
        "rationale": "",
    }

    if dry_run:
        row.update(
            {
                "ok": True,
                "prompt_preview": prompt[:2500],
                "rationale": "Dry run only. LLM was not called.",
            }
        )
        return row

    from app.services.llm_service import LLMService

    llm = LLMService(temperature=0.0, max_tokens=1200)
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    data = parse_json_response(response)
    row.update(
        {
            "ok": True,
            "scores": normalize_scores(data),
            "major_issues": data.get("major_issues") if isinstance(data.get("major_issues"), list) else [],
            "rationale": str(data.get("rationale") or ""),
        }
    )
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    summary: dict[str, Any] = {
        "total": len(rows),
        "ok_rate": round(len(ok_rows) / len(rows), 4) if rows else 0.0,
    }
    for key in SCORE_KEYS:
        values = [float(row.get("scores", {}).get(key, 0.0)) for row in ok_rows]
        summary[f"{key}_avg"] = round(sum(values) / len(values), 3) if values else 0.0
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run optional LLM-as-Judge evaluation for TripPlan outputs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="TripPlan JSON/JSONL path.")
    source.add_argument("--trip-id", help="Read a stored trip from Redis by trip id.")
    parser.add_argument("--output", help="Optional report JSON path.")
    parser.add_argument("--dry-run", action="store_true", help="Build judge prompts without calling the LLM.")
    args = parser.parse_args()

    plans = load_trip_from_redis(args.trip_id) if args.trip_id else load_input(Path(args.input))
    rows = []
    for index, row in enumerate(plans):
        plan = extract_trip_payload(row)
        try:
            rows.append(judge_one(plan, index, args.dry_run))
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "index": index,
                    "ok": False,
                    "dry_run": args.dry_run,
                    "scores": {key: 0.0 for key in SCORE_KEYS},
                    "major_issues": [str(exc)],
                    "rationale": "Judge evaluation failed.",
                }
            )

    report = {
        "summary": summarize(rows),
        "records": rows,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
