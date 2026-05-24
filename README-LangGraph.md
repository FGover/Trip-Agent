# WayfinderAI LangGraph Refactor

This directory is the LangChain / LangGraph refactor of `WayfinderAI`.

## Migrated Features

- Vue 3 frontend: login, register, profile, trip creation, trip detail, edit, version rollback, delete, export.
- FastAPI backend API remains compatible with the original frontend.
- Redis stores users, guest sessions, async tasks, trips, and trip versions.
- Alibaba Cloud DashScope OpenAI-compatible LLM calls.
- Alibaba Cloud DashScope `text-embedding-v4` embeddings.
- FAISS vector memory for user preferences and trip memory.
- AMap attraction, hotel, and weather tools.
- Unsplash attraction image enrichment.

## Tech Stack

- Frontend: Vue 3, Vite, TypeScript, Element Plus, Pinia.
- Backend: FastAPI, Pydantic, Redis, FAISS.
- Agent workflow: LangChain + LangGraph.
- Tool protocol: MCP stdio tools for AMap POI and weather queries.
- Retrieval: Hybrid RAG with FAISS semantic retrieval and Okapi BM25 lexical retrieval.
- Model provider: OpenAI-compatible DashScope endpoints.

## LangGraph Workflow

```text
load_memory
  -> query_attractions
  -> query_hotels
  -> query_weather
  -> generate_plan
```

The query nodes call MCP tools first. If MCP is unavailable, the backend falls back to the local AMap HTTP implementation.

## Hybrid RAG And Reranking

The memory layer combines:

- FAISS semantic retrieval with `text-embedding-v4`, normalized vectors, and `IndexFlatIP`.
- Okapi BM25 lexical retrieval with Chinese n-gram tokenization.

The retrieval layer then applies dynamic vector/BM25 weights, RRF ranking fusion, and lightweight business reranking based on lexical overlap, memory freshness, and memory type. The frontend extra requirements field is included in the retrieval query and persisted into user memory.

## MCP Tools

The MCP server is implemented in:

```text
backend/app/mcp/amap_server.py
```

It exposes:

- `search_pois`: AMap POI search for attractions, hotels, restaurants, and other places.
- `get_weather_forecast`: AMap city weather forecast.

The LangGraph planner calls these tools through:

```text
backend/app/services/mcp_tool_service.py
```

## Start Redis

```powershell
docker run -d --name wayfinder-redis -p 6379:6379 redis:7
```

If the container already exists:

```powershell
docker start wayfinder-redis
```

## Start Backend

```powershell
cd E:\javaLearning\Ai\WayfinderAI-LangGraph\backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python run.py
```

## Start Frontend

```powershell
cd E:\javaLearning\Ai\WayfinderAI-LangGraph\frontend
npm install
npm run dev
```
