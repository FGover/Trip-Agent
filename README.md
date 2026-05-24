# WayfinderAI LangGraph

This is the LangChain / LangGraph refactor of `WayfinderAI`. The original project remains unchanged. This version keeps the frontend and API behavior compatible while replacing the original agent orchestration with LangGraph.

## Tech Stack

- Frontend: Vue 3, TypeScript, Vite, Pinia, Vue Router, Element Plus, Axios, AMap JS API.
- Backend: FastAPI, Uvicorn, Pydantic, Redis, FAISS.
- Agent workflow: LangChain, LangGraph, langchain-openai.
- Tool protocol: MCP stdio tools for AMap POI and weather queries.
- Retrieval: Hybrid RAG with FAISS semantic retrieval and Okapi BM25 lexical retrieval.
- Models: Alibaba Cloud DashScope OpenAI-compatible LLM and `text-embedding-v4`.
- External services: AMap API and Unsplash API.

## Migrated Features

- User registration, login, guest session, profile update, avatar upload, password change.
- Structured travel preferences plus natural-language extra requirements.
- Async trip generation with task progress polling.
- Attraction, hotel, and weather queries through MCP tools.
- Trip detail, my trips, edit, delete, version history, and rollback.
- User preference memory, historical trip memory, and vector retrieval.
- Image enrichment, budget breakdown, map display, and PDF/image export.

## Core Workflow

Main implementation:

```text
backend/app/agents/planner.py
```

LangGraph workflow:

```text
load_memory -> query_attractions -> query_hotels -> query_weather -> generate_plan
```

The query nodes call MCP tools first. If MCP is unavailable, the backend falls back to the local AMap HTTP implementation.

## Hybrid RAG And Reranking

The memory layer combines two retrieval signals:

- FAISS vector retrieval: `text-embedding-v4` embeddings, normalized vectors, `IndexFlatIP`.
- BM25 lexical retrieval: Okapi BM25 with Chinese n-gram tokenization for exact city, attraction, hotel, and preference terms.

The memory node applies:

- Dynamic vector/BM25 weights based on query specificity and natural-language constraints.
- RRF ranking fusion across FAISS and BM25 candidate lists.
- Lightweight business reranking using lexical overlap, memory freshness, and memory type.

The top reranked memories are injected into the final planning prompt. The natural-language extra requirements field is included in the retrieval query and stored back into user memory after generation.

MCP implementation:

```text
backend/app/mcp/amap_server.py
backend/app/services/mcp_tool_service.py
```

## Start

Start Redis:

```powershell
docker start wayfinder-redis
```

If the container does not exist:

```powershell
docker run -d --name wayfinder-redis -p 6379:6379 redis:7
```

Start backend:

```powershell
cd E:\javaLearning\Ai\WayfinderAI-LangGraph\backend
.\.venv\Scripts\python.exe run.py
```

Start frontend:

```powershell
cd E:\javaLearning\Ai\WayfinderAI-LangGraph\frontend
npm run dev
```

Open:

```text
http://localhost:5173
```
