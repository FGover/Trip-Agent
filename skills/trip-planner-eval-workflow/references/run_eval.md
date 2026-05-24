# Running Evaluation

## Rule Evaluation From A JSON File

```powershell
cd E:\javaLearning\Ai\WayfinderAI-LangGraph\backend
.\.venv\Scripts\python.exe tools\evaluate_trip_plan.py --input tests\fixtures\sample_trip_plan.json --output reports\sample_trip_eval_report.json
```

Use this for saved plans, test fixtures, and regression snapshots.

## Rule Evaluation From Redis

```powershell
cd E:\javaLearning\Ai\WayfinderAI-LangGraph\backend
.\.venv\Scripts\python.exe tools\evaluate_trip_plan.py --trip-id <trip_id> --output reports\trip_eval_report.json
```

Use this after the web app generates a real trip and stores it in Redis.

## LLM-as-Judge Dry Run

```powershell
cd E:\javaLearning\Ai\WayfinderAI-LangGraph\backend
.\.venv\Scripts\python.exe tools\judge_trip_plan.py --input tests\fixtures\sample_trip_plan.json --dry-run --output reports\sample_judge_prompt.json
```

Dry run does not call the LLM. It only checks that the judge prompt and input payload can be built.

## LLM-as-Judge Real Run

```powershell
cd E:\javaLearning\Ai\WayfinderAI-LangGraph\backend
.\.venv\Scripts\python.exe tools\judge_trip_plan.py --input tests\fixtures\sample_trip_plan.json --output reports\sample_judge_report.json
```

Use this only when LLM credentials are configured and cost/time are acceptable.

