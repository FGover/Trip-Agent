# Metrics

## Rule Metrics

- `schema_ok_rate`: TripPlan JSON can be parsed by the backend Pydantic model.
- `hard_pass_rate`: required structure checks passed, including title, non-empty days, day order, attractions, dining, and attraction count.
- `soft_pass_rate`: hard checks passed and most quality checks are acceptable.
- `budget_consistency_rate`: total budget is close to the sum of transport, dining, hotel, and ticket costs.
- `daily_budget_rollup_consistency_rate`: whole-trip budget fields match the sum of daily budget fields.
- `attraction_location_rate`: attractions with valid latitude and longitude.
- `hotel_location_rate`: hotels with valid latitude and longitude.
- `dining_location_rate`: dining records with valid latitude and longitude.

## Judge Metrics

LLM-as-Judge scores are optional and should be used after rule checks.

- `preference_satisfaction`: whether the plan reflects user preferences and constraints.
- `practicality`: whether the route density and daily schedule are realistic.
- `grounding_faithfulness`: whether the plan is faithful to retrieved/tool-provided facts.
- `budget_reasonableness`: whether the budget is reasonable and internally consistent.
- `coherence`: whether the whole itinerary is coherent and not repetitive.
- `overall_quality`: final subjective quality score.

## Interview Explanation

Rule metrics answer "can this plan be safely rendered and used?" Judge metrics answer "is this a good travel plan?"
