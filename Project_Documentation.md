# WayfinderAI LangGraph 重构说明

本项目是项目二 `WayfinderAI` 的技术栈重构版，目标是在功能完整迁移的前提下，将原 hello-agents 多智能体实现替换为更常见的 LangChain / LangGraph 技术栈。

## 架构概览

```text
Vue3 前端
  -> FastAPI API
  -> Redis 异步任务队列
  -> LangGraph 行程规划工作流
  -> 高德地图 / 百炼 LLM / 百炼 Embedding / Unsplash
  -> Redis + FAISS 存储结果和记忆
```

## LangGraph 节点

- `load_memory`：读取用户历史偏好、历史行程和知识记忆。
- `query_attractions`：通过高德地图文本搜索查询景点候选。
- `query_hotels`：通过高德地图文本搜索查询酒店候选。
- `query_weather`：通过高德天气接口查询天气。
- `generate_plan`：通过 LangChain 的 OpenAI 兼容模型生成结构化行程 JSON。

## 兼容性

前端接口协议保持与项目二一致：

- `/api/v1/auth/*`
- `/api/v1/trips/plan-async`
- `/api/v1/trips/tasks/{task_id}`
- `/api/v1/trips/list`
- `/api/v1/trips/{trip_id}`
- `/api/v1/trips/{trip_id}/versions`
- `/api/v1/trips/{trip_id}/rollback`

因此前端功能不需要重新设计。
