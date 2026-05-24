"""Rule-based evaluator for WayfinderAI TripPlan outputs.

The evaluator is intentionally deterministic and offline. It validates TripPlan
shape with Pydantic and reports practical quality signals that can be used before
running slower LLM-as-Judge checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.trip_model import TripPlanResponse  # noqa: E402


@dataclass
class EvalAccumulator:
    total: int = 0
    schema_ok: int = 0
    hard_pass: int = 0
    soft_pass: int = 0
    attraction_total: int = 0
    attraction_with_location: int = 0
    hotel_total: int = 0
    hotel_with_location: int = 0
    dining_total: int = 0
    dining_with_location: int = 0
    budget_consistent: int = 0
    daily_budget_rollup_consistent: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)


def load_input(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            rows.append(extract_trip_payload(row))
        return rows

    data = json.loads(text)
    if isinstance(data, list):
        return [extract_trip_payload(item) for item in data]
    if isinstance(data, dict):
        return [extract_trip_payload(data)]
    raise ValueError("Input must be a JSON object, JSON array, or JSONL file")


def load_trip_from_redis(trip_id: str) -> list[dict[str, Any]]:
    from app.services.redis_service import redis_service

    trip = redis_service.get_trip(trip_id)
    if not trip:
        raise ValueError(f"Trip not found in Redis: {trip_id}")
    return [trip]


def extract_trip_payload(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("trip", "plan", "result", "data"):
        value = row.get(key)
        if isinstance(value, dict) and "days" in value:
            return value
    return row


def has_valid_location(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, dict):
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        else:
            return False
    lat = value.get("lat")
    lng = value.get("lng")
    return isinstance(lat, (int, float)) and isinstance(lng, (int, float))


def numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.replace("元", "").replace("/晚", "").strip()
        try:
            return float(stripped)
        except ValueError:
            return 0.0
    return 0.0


def evaluate_one(raw: dict[str, Any], index: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "index": index,
        "schema_ok": False,
        "hard_pass": False,
        "soft_pass": False,
        "metrics": {},
        "errors": [],
        "warnings": [],
    }

    try:
        plan = TripPlanResponse.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append({"stage": "schema", "message": str(exc)})
        return result

    result["schema_ok"] = True
    data = plan.model_dump()
    metrics = result["metrics"]

    days = data.get("days") or []
    hotels = data.get("hotels") or []
    total_budget = data.get("total_budget") or {}

    metrics["has_title"] = bool(data.get("trip_title"))
    metrics["days_count"] = len(days)
    metrics["days_non_empty"] = len(days) > 0
    metrics["day_index_ok"] = all(day.get("day") == i + 1 for i, day in enumerate(days))
    metrics["daily_theme_ok"] = all(bool(day.get("theme")) for day in days)
    metrics["daily_weather_present"] = all(day.get("weather") is not None for day in days)
    metrics["daily_attractions_present"] = all(len(day.get("attractions") or []) > 0 for day in days)
    metrics["daily_attraction_count_ok"] = all(1 <= len(day.get("attractions") or []) <= 4 for day in days)
    metrics["daily_dining_present"] = all(len(day.get("dinings") or []) > 0 for day in days)
    metrics["hotel_candidates_present"] = len(hotels) > 0

    attraction_total = 0
    attraction_with_location = 0
    dining_total = 0
    dining_with_location = 0
    recommended_hotel_total = 0
    recommended_hotel_with_location = 0

    for day in days:
        for attraction in day.get("attractions") or []:
            attraction_total += 1
            if has_valid_location(attraction.get("location")):
                attraction_with_location += 1
            if not attraction.get("name"):
                result["warnings"].append({"stage": "attraction", "message": "Attraction without name"})

        for dining in day.get("dinings") or []:
            dining_total += 1
            if has_valid_location(dining.get("location")):
                dining_with_location += 1
            if not dining.get("name"):
                result["warnings"].append({"stage": "dining", "message": "Dining without name"})

        hotel = day.get("recommended_hotel")
        if hotel:
            recommended_hotel_total += 1
            if has_valid_location(hotel.get("location")):
                recommended_hotel_with_location += 1

    hotel_total = len(hotels) + recommended_hotel_total
    hotel_with_location = sum(1 for hotel in hotels if has_valid_location(hotel.get("location")))
    hotel_with_location += recommended_hotel_with_location

    metrics["attraction_total"] = attraction_total
    metrics["attraction_location_rate"] = rate(attraction_with_location, attraction_total)
    metrics["dining_total"] = dining_total
    metrics["dining_location_rate"] = rate(dining_with_location, dining_total)
    metrics["hotel_total"] = hotel_total
    metrics["hotel_location_rate"] = rate(hotel_with_location, hotel_total)

    budget_parts = (
        numeric(total_budget.get("transport_cost"))
        + numeric(total_budget.get("dining_cost"))
        + numeric(total_budget.get("hotel_cost"))
        + numeric(total_budget.get("attraction_ticket_cost"))
    )
    total = numeric(total_budget.get("total"))
    budget_delta = abs(total - budget_parts)
    metrics["budget_total"] = total
    metrics["budget_parts_sum"] = budget_parts
    metrics["budget_delta"] = round(budget_delta, 2)
    metrics["budget_consistent"] = total == 0 or budget_delta <= max(5.0, total * 0.08)

    daily_budget_rollup = {
        "transport_cost": 0.0,
        "dining_cost": 0.0,
        "hotel_cost": 0.0,
        "attraction_ticket_cost": 0.0,
        "total": 0.0,
    }
    daily_budget_issues: list[dict[str, Any]] = []
    for day in days:
        day_budget = day.get("budget") or {}
        day_parts = (
            numeric(day_budget.get("transport_cost"))
            + numeric(day_budget.get("dining_cost"))
            + numeric(day_budget.get("hotel_cost"))
            + numeric(day_budget.get("attraction_ticket_cost"))
        )
        day_total = numeric(day_budget.get("total"))
        if day_total and abs(day_total - day_parts) > max(5.0, day_total * 0.08):
            daily_budget_issues.append(
                {
                    "day": day.get("day"),
                    "issue": "daily_total_mismatch",
                    "total": day_total,
                    "parts_sum": round(day_parts, 2),
                    "delta": round(abs(day_total - day_parts), 2),
                }
            )
        for key in daily_budget_rollup:
            daily_budget_rollup[key] += numeric(day_budget.get(key))

    rollup_issues: list[dict[str, Any]] = []
    for key, daily_value in daily_budget_rollup.items():
        whole_value = numeric(total_budget.get(key))
        tolerance_base = max(abs(whole_value), abs(daily_value), 1.0)
        if abs(whole_value - daily_value) > max(5.0, tolerance_base * 0.08):
            rollup_issues.append(
                {
                    "field": key,
                    "total_budget_value": round(whole_value, 2),
                    "daily_rollup_value": round(daily_value, 2),
                    "delta": round(abs(whole_value - daily_value), 2),
                }
            )

    metrics["daily_budget_rollup"] = {key: round(value, 2) for key, value in daily_budget_rollup.items()}
    metrics["daily_budget_issues"] = daily_budget_issues
    metrics["budget_rollup_issues"] = rollup_issues
    metrics["daily_budget_rollup_consistent"] = not daily_budget_issues and not rollup_issues

    hard_checks = [
        metrics["has_title"],
        metrics["days_non_empty"],
        metrics["day_index_ok"],
        metrics["daily_attractions_present"],
        metrics["daily_dining_present"],
        metrics["daily_attraction_count_ok"],
    ]
    soft_checks = [
        metrics["daily_theme_ok"],
        metrics["daily_weather_present"],
        metrics["hotel_candidates_present"],
        metrics["budget_consistent"],
        metrics["daily_budget_rollup_consistent"],
        metrics["attraction_location_rate"] >= 0.8 if attraction_total else False,
    ]

    result["hard_pass"] = all(hard_checks)
    result["soft_pass"] = result["hard_pass"] and sum(bool(item) for item in soft_checks) >= 3

    if not result["hard_pass"]:
        result["errors"].append({"stage": "hard_checks", "message": "One or more required structure checks failed"})
    if not result["soft_pass"]:
        result["warnings"].append({"stage": "soft_checks", "message": "One or more quality checks need attention"})
    if daily_budget_issues:
        result["warnings"].append(
            {
                "stage": "daily_budget",
                "message": "Daily budget totals do not match their budget parts",
                "details": daily_budget_issues,
            }
        )
    if rollup_issues:
        result["warnings"].append(
            {
                "stage": "budget_rollup",
                "message": "Whole-trip budget fields do not match the sum of daily budgets",
                "details": rollup_issues,
            }
        )

    return result


def rate(hit: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(hit / total, 4)


def summarize(acc: EvalAccumulator) -> dict[str, Any]:
    return {
        "total": acc.total,
        "schema_ok_rate": rate(acc.schema_ok, acc.total),
        "hard_pass_rate": rate(acc.hard_pass, acc.total),
        "soft_pass_rate": rate(acc.soft_pass, acc.total),
        "budget_consistency_rate": rate(acc.budget_consistent, acc.schema_ok),
        "daily_budget_rollup_consistency_rate": rate(acc.daily_budget_rollup_consistent, acc.schema_ok),
        "attraction_location_rate": rate(acc.attraction_with_location, acc.attraction_total),
        "hotel_location_rate": rate(acc.hotel_with_location, acc.hotel_total),
        "dining_location_rate": rate(acc.dining_with_location, acc.dining_total),
    }


def evaluate(plans: list[dict[str, Any]]) -> dict[str, Any]:
    acc = EvalAccumulator(total=len(plans))
    for index, raw in enumerate(plans):
        row = evaluate_one(raw, index)
        acc.records.append(row)
        if row["schema_ok"]:
            acc.schema_ok += 1
        if row["hard_pass"]:
            acc.hard_pass += 1
        if row["soft_pass"]:
            acc.soft_pass += 1

        metrics = row.get("metrics") or {}
        if metrics.get("budget_consistent"):
            acc.budget_consistent += 1
        if metrics.get("daily_budget_rollup_consistent"):
            acc.daily_budget_rollup_consistent += 1
        acc.attraction_total += int(metrics.get("attraction_total") or 0)
        acc.attraction_with_location += round((metrics.get("attraction_location_rate") or 0) * int(metrics.get("attraction_total") or 0))
        acc.hotel_total += int(metrics.get("hotel_total") or 0)
        acc.hotel_with_location += round((metrics.get("hotel_location_rate") or 0) * int(metrics.get("hotel_total") or 0))
        acc.dining_total += int(metrics.get("dining_total") or 0)
        acc.dining_with_location += round((metrics.get("dining_location_rate") or 0) * int(metrics.get("dining_total") or 0))

    return {
        "summary": summarize(acc),
        "records": acc.records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate WayfinderAI TripPlan JSON outputs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="TripPlan JSON/JSONL path.")
    source.add_argument("--trip-id", help="Read a stored trip from Redis by trip id.")
    parser.add_argument("--output", help="Optional report JSON path.")
    args = parser.parse_args()

    if args.trip_id:
        plans = load_trip_from_redis(args.trip_id)
    else:
        input_path = Path(args.input)
        plans = load_input(input_path)
    report = evaluate(plans)
    text = json.dumps(report, ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
