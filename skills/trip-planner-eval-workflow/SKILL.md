---
name: trip-planner-eval-workflow
description: Use this skill when evaluating WayfinderAI trip-planning outputs, checking generated TripPlan JSON quality, comparing planner behavior, or preparing resume/project evidence for rule-based and judge-based evaluation.
---

# Trip Planner Eval Workflow

## Purpose

This skill defines the project-level workflow for evaluating generated travel plans. It is not a user-facing product feature. It is a reusable development workflow that helps the project check whether planner outputs are structurally valid, grounded in retrieved candidates, practical, and ready for frontend rendering.

## When To Use

Use this skill when you need to:

- Validate generated trip plan JSON before demos or regression checks.
- Compare planner changes after modifying prompts, retrieval, MCP tools, or reranking.
- Explain how the project evaluates plan quality in interviews.
- Produce a compact evaluation report for a set of saved plans.

## Evaluation Layers

### Rule Evaluation

Run deterministic checks first. These checks are cheap, stable, and do not call an LLM.

- JSON and Pydantic schema validity.
- Trip shape: city title, days list, daily weather, hotels, attractions, dining, and budgets.
- Daily structure: each day should have a valid day index, theme, attractions, dining, and daily budget.
- Location validity: locations should contain numeric `lat` and `lng` when present.
- Grounding reference: if candidate snapshots are provided, attractions, hotels, and dining can be matched against candidate names.
- Budget consistency: total budget should be close to the sum of transport, dining, hotel, and attraction ticket costs.
- Frontend readiness: fields required by Result/Edit/MyTrips pages should be present.

### LLM-as-Judge Evaluation

Use this only after rule evaluation passes. Judge evaluation is for quality, not basic validity.

Suggested judge dimensions:

- Preference satisfaction.
- Practicality.
- Grounding faithfulness.
- Budget reasonableness.
- Coherence.
- Overall quality.

For this project, judge evaluation should use the same TripPlan, user request, and compact planner context to avoid scoring from incomplete context.

## Default Workflow

1. Save generated plans as JSON files or JSONL rows.
2. Run the rule evaluator:

```powershell
cd E:\javaLearning\Ai\WayfinderAI-LangGraph\backend
.\.venv\Scripts\python.exe tools\evaluate_trip_plan.py --input path\to\trip.json --output reports\trip_eval_report.json
```

3. Inspect the report summary:

- `schema_ok`
- `hard_pass`
- `soft_pass`
- grounding rates
- budget consistency
- warnings and errors

4. If rule checks pass but quality is still uncertain, run an LLM-as-Judge pass separately.
5. When comparing two planner versions, use the same input requests and candidate context for both runs.

## References

- `references/run_eval.md`: commands for file-based, Redis-based, and judge-based evaluation.
- `references/metrics.md`: metric definitions and how to explain results.
- `references/reporting.md`: suggested reporting format and resume-safe wording.

## Resume-Safe Description

You can describe this workflow as:

"构建项目级 Skill 工作流，规范 Planner 数据生成、规则评估与 LLM-as-Judge 评估流程，从结构合法性、工具结果忠实度、预算合理性和偏好满足度等维度支持生成效果迭代。"
