from __future__ import annotations

import asyncio
import json
import math
import re
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, TypedDict

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.models.common_model import Attraction, Dining, Hotel, Location, Weather
from app.models.trip_model import BudgetBreakdown, DailyBudget, DailyPlan, TripPlanRequest, TripPlanResponse
from app.observability.logger import default_logger as logger
from app.observability.logger import get_request_id
from app.services.city_service import city_support_service
from app.services.llm_service import LLMService
from app.services.mcp_tool_service import amap_mcp_client
from app.services.rerank_service import rerank_service
from app.services.unsplash_service import UnsplashService
from app.services.vector_memory_service import VectorMemoryService


CITY_BOUNDS = {
    "北京": {"lat_min": 39.4, "lat_max": 41.1, "lng_min": 115.7, "lng_max": 117.4},
    "上海": {"lat_min": 30.7, "lat_max": 31.9, "lng_min": 120.8, "lng_max": 122.2},
    "广州": {"lat_min": 22.7, "lat_max": 23.8, "lng_min": 112.9, "lng_max": 114.0},
    "深圳": {"lat_min": 22.4, "lat_max": 22.9, "lng_min": 113.7, "lng_max": 114.6},
    "成都": {"lat_min": 30.4, "lat_max": 30.9, "lng_min": 103.9, "lng_max": 104.5},
    "杭州": {"lat_min": 30.0, "lat_max": 30.5, "lng_min": 119.5, "lng_max": 120.5},
    "重庆": {"lat_min": 29.3, "lat_max": 29.9, "lng_min": 106.2, "lng_max": 106.8},
    "武汉": {"lat_min": 30.3, "lat_max": 31.0, "lng_min": 113.9, "lng_max": 114.6},
    "西安": {"lat_min": 34.0, "lat_max": 34.5, "lng_min": 108.7, "lng_max": 109.2},
    "苏州": {"lat_min": 31.1, "lat_max": 31.5, "lng_min": 120.3, "lng_max": 121.0},
    "天津": {"lat_min": 38.9, "lat_max": 39.6, "lng_min": 116.9, "lng_max": 117.9},
    "南京": {"lat_min": 31.9, "lat_max": 32.2, "lng_min": 118.4, "lng_max": 119.2},
    "长沙": {"lat_min": 28.1, "lat_max": 28.4, "lng_min": 112.8, "lng_max": 113.2},
    "郑州": {"lat_min": 34.4, "lat_max": 34.9, "lng_min": 113.4, "lng_max": 113.9},
    "厦门": {"lat_min": 24.4, "lat_max": 24.6, "lng_min": 118.0, "lng_max": 118.2},
    "青岛": {"lat_min": 35.9, "lat_max": 36.4, "lng_min": 119.9, "lng_max": 120.7},
    "大连": {"lat_min": 38.7, "lat_max": 39.2, "lng_min": 121.3, "lng_max": 122.0},
    "三亚": {"lat_min": 18.1, "lat_max": 18.4, "lng_min": 109.3, "lng_max": 109.7},
    "丽江": {"lat_min": 26.8, "lat_max": 27.2, "lng_min": 100.1, "lng_max": 100.5},
    "桂林": {"lat_min": 24.2, "lat_max": 26.4, "lng_min": 109.6, "lng_max": 111.5},
    "昆明": {"lat_min": 24.7, "lat_max": 25.3, "lng_min": 102.5, "lng_max": 103.1},
    "哈尔滨": {"lat_min": 45.5, "lat_max": 46.0, "lng_min": 126.4, "lng_max": 127.1},
    "沈阳": {"lat_min": 41.5, "lat_max": 42.0, "lng_min": 123.2, "lng_max": 123.8},
    "济南": {"lat_min": 36.5, "lat_max": 36.8, "lng_min": 116.8, "lng_max": 117.3},
    "黄山": {"lat_min": 29.8, "lat_max": 30.2, "lng_min": 118.1, "lng_max": 118.5},
    "张家界": {"lat_min": 28.9, "lat_max": 29.3, "lng_min": 110.2, "lng_max": 110.7},
    "敦煌": {"lat_min": 39.8, "lat_max": 40.3, "lng_min": 94.4, "lng_max": 95.1},
    "拉萨": {"lat_min": 29.5, "lat_max": 30.0, "lng_min": 90.9, "lng_max": 91.5},
    "乌鲁木齐": {"lat_min": 43.7, "lat_max": 44.2, "lng_min": 87.4, "lng_max": 88.0},
    "宁波": {"lat_min": 29.8, "lat_max": 30.0, "lng_min": 121.3, "lng_max": 121.8},
}

TRAVEL_INTENT_KEYWORDS = {
    "photography": ("摄影", "拍照", "打卡", "出片", "照片", "网红", "机位"),
    "nature": ("自然", "风景", "公园", "湿地", "海边", "滨海", "生态"),
    "family": ("亲子", "儿童", "孩子", "小孩", "科普"),
    "low_walking": ("不想走太多路", "少走路", "少步行", "不累", "轻松", "老人", "带娃"),
    "transit_hotel": ("靠近地铁", "地铁", "交通方便", "交通便利", "近地铁"),
    "water": ("玩水", "戏水", "漂流", "竹筏", "游船", "水上", "亲水", "江", "河", "湖"),
    "night_view": ("夜景", "夜游", "夜市", "灯光", "晚上", "夜晚"),
}

FALLBACK_INTENT_SEARCH_TERMS = {
    "photography": ("网红打卡", "热门打卡点", "城市地标", "观景台", "特色街区", "滨水景观", "城市景观", "热门景区"),
    "nature": ("自然 公园", "滨海 公园"),
    "family": ("亲子 公园", "儿童 科普"),
    "low_walking": ("公园 休闲", "城市公园"),
    "water": ("玩水 漂流", "竹筏 游船", "亲水 景区"),
    "night_view": ("夜景", "夜游 灯光", "夜市 美食"),
}

DEFAULT_QUERY_BUNDLE = {
    "memory_queries": [],
    "attraction_queries": [],
    "dining_queries": [],
    "hotel_queries": [],
    "must_have": [],
    "prefer": [],
    "avoid": [],
    "time_constraints": [],
    "location_constraints": [],
}

TRAVEL_INTENT_RERANK_TEXT = {
    "photography": "旅游摄影打卡: 选择有代表性、热门、出片、适合拍照留念的旅游地点，如网红打卡点、观景台、城市地标、特色街区、历史街区、滨水景观、公园或热门景区",
    "nature": "自然风光、公园、湿地、滨海或生态体验",
    "family": "亲子友好、儿童可参与、科普或轻松游玩",
    "low_walking": "不想走太多路，避免登山、长徒步、体力消耗大",
    "transit_hotel": "酒店靠近地铁或交通便利",
    "water": "想玩水，优先匹配漂流、竹筏、游船、亲水景区、江河湖相关体验",
    "night_view": "想看夜景，优先匹配夜游、灯光、两江四湖、夜市或适合夜间游览的地点",
}

VISIT_POI_TYPE_CATEGORIES = (
    "风景名胜",
    "公园",
    "广场",
    "博物馆",
    "纪念馆",
    "展览馆",
    "体育休闲",
    "科教文化",
)

NON_VISIT_POI_TYPE_CATEGORIES = (
    "住宿服务",
    "宾馆酒店",
    "商务住宅",
    "生活服务",
    "公司企业",
    "购物服务",
    "医疗保健服务",
)


class PlannerState(TypedDict, total=False):
    request: TripPlanRequest
    user_id: str
    request_id: str
    progress_callback: Optional[Callable[[int, str], None]]
    user_memories: List[Dict[str, Any]]
    knowledge_memories: List[Dict[str, Any]]
    requirement_constraints: Dict[str, Any]
    attractions: Dict[str, Any]
    hotels: Dict[str, Any]
    dinings: Dict[str, Any]
    weather: Dict[str, Any]
    plan: TripPlanResponse


class PlannerAgent:
    """LangGraph based trip planner.

    This class preserves the public contract of the original hello-agents based
    orchestrator while replacing the orchestration engine with LangGraph nodes.
    """

    def __init__(self, llm_service: LLMService, memory_service: VectorMemoryService = None):
        self.llm = LLMService()
        self.memory_service = memory_service or VectorMemoryService()
        self.unsplash_service = UnsplashService(settings.UNSPLASH_ACCESS_KEY)
        self._regeo_cache: Dict[str, Dict[str, str]] = {}
        self.chat_model = ChatOpenAI(
            model=settings.LLM_MODEL_ID or "gpt-4-turbo",
            api_key=settings.LLM_API_KEY or settings.OPENAI_API_KEY,
            base_url=settings.LLM_BASE_URL,
            temperature=0.4,
            max_tokens=4096,
            timeout=settings.LLM_TIMEOUT,
            extra_body={"enable_thinking": False},
        )
        self.graph = self._build_graph()
        logger.info("LangGraph trip planner initialized")

    def _build_graph(self):
        graph = StateGraph(PlannerState)
        graph.add_node("load_memory", self._load_memory_node)
        graph.add_node("query_attractions", self._query_attractions_node)
        graph.add_node("query_hotels", self._query_hotels_node)
        graph.add_node("query_dinings", self._query_dinings_node)
        graph.add_node("query_weather", self._query_weather_node)
        graph.add_node("generate_plan", self._generate_plan_node)
        graph.add_edge(START, "load_memory")
        graph.add_edge("load_memory", "query_attractions")
        graph.add_edge("query_attractions", "query_hotels")
        graph.add_edge("query_hotels", "query_dinings")
        graph.add_edge("query_dinings", "query_weather")
        graph.add_edge("query_weather", "generate_plan")
        graph.add_edge("generate_plan", END)
        return graph.compile()

    def _report(self, state: PlannerState, progress: int, message: str) -> None:
        callback = state.get("progress_callback")
        if not callback:
            return
        try:
            callback(progress, message)
        except Exception as exc:
            logger.warning("Failed to update trip task progress", extra={"error": str(exc)})

    def _load_memory_node(self, state: PlannerState) -> Dict[str, Any]:
        request = state["request"]
        user_id = state["user_id"]
        requirement_constraints = self._analyze_requirement_constraints(request)
        query_bundle = requirement_constraints.get("query_bundle", {}) if isinstance(requirement_constraints, dict) else {}
        original_memory_query = self._build_original_memory_query(request)
        expanded_memory_queries = self._build_expanded_memory_queries(request, query_bundle)
        user_memories = self._retrieve_user_memories_three_way(
            user_id,
            original_query=original_memory_query,
            expanded_queries=expanded_memory_queries,
            limit=5,
        )
        knowledge_memories = self._retrieve_knowledge_memories_three_way(
            original_query=original_memory_query,
            expanded_queries=expanded_memory_queries,
            limit=3,
        )
        self._report(state, 30, "Querying attractions, hotels, and weather")
        return {
            "user_memories": user_memories or [],
            "knowledge_memories": knowledge_memories or [],
            "requirement_constraints": requirement_constraints,
        }

    def _query_attractions_node(self, state: PlannerState) -> Dict[str, Any]:
        request = state["request"]
        requirement_constraints = state.get("requirement_constraints", {})
        items: List[Dict[str, Any]] = []
        for keyword in self._build_attraction_search_terms(request, requirement_constraints):
            items.extend(
                self._amap_text_search(
                    keywords=keyword,
                    city=request.destination,
                    limit=6,
                    category="景点",
                )
            )
        items = self._rank_pois_for_request(
            self._dedupe_pois(items),
            request=request,
            constraints=requirement_constraints,
            poi_kind="attraction",
        )[:10]
        payload = {
            "summary": f"高德地图返回 {len(items)} 个候选景点",
            "warnings": [] if items else ["no_attractions_found"],
            "items": items,
        }
        self._report(state, 45, "Attractions query completed")
        return {"attractions": payload}

    def _query_hotels_node(self, state: PlannerState) -> Dict[str, Any]:
        request = state["request"]
        requirement_constraints = state.get("requirement_constraints", {})
        intents = requirement_constraints.get("intents", {}) if isinstance(requirement_constraints, dict) else {}
        query_bundle = requirement_constraints.get("query_bundle", {}) if isinstance(requirement_constraints, dict) else {}
        hotel_terms = self._normalize_query_list(query_bundle.get("hotel_queries"), request.destination)
        if not hotel_terms:
            keyword = request.hotel_preferences[0] if request.hotel_preferences else "酒店"
            if intents.get("transit_hotel"):
                keyword = "地铁 酒店"
            hotel_terms.append(keyword if "酒店" in keyword else f"{keyword} 酒店")
        items: List[Dict[str, Any]] = []
        for keyword in self._dedupe_texts(hotel_terms)[:5]:
            items.extend(
                self._amap_text_search(
                    keywords=keyword if "酒店" in keyword else f"{keyword} 酒店",
                    city=request.destination,
                    limit=5,
                    category="酒店",
                )
            )
        if not items:
            fallback_keyword = request.hotel_preferences[0] if request.hotel_preferences else "酒店"
            items = self._amap_text_search(
                keywords=fallback_keyword if "酒店" in fallback_keyword else f"{fallback_keyword} 酒店",
                city=request.destination,
                limit=5,
                category="酒店",
            )
        items = self._rank_pois_for_request(
            self._dedupe_pois(items),
            request=request,
            constraints=requirement_constraints,
            poi_kind="hotel",
        )[:5]
        payload = {
            "summary": f"高德地图返回 {len(items)} 个候选酒店",
            "warnings": [] if items else ["no_hotels_found"],
            "items": items,
        }
        self._report(state, 55, "Hotels query completed")
        return {"hotels": payload}

    def _query_dinings_node(self, state: PlannerState) -> Dict[str, Any]:
        request = state["request"]
        requirement_constraints = state.get("requirement_constraints", {})
        terms = self._build_dining_search_terms(request, requirement_constraints)
        items: List[Dict[str, Any]] = []
        for keyword in terms:
            items.extend(
                self._amap_text_search(
                    keywords=keyword,
                    city=request.destination,
                    limit=5,
                    category="餐饮",
                )
            )
        items = self._rank_pois_for_request(
            self._dedupe_pois(items),
            request=request,
            constraints=requirement_constraints,
            poi_kind="dining",
        )[:10]
        payload = {
            "summary": f"高德地图返回 {len(items)} 个候选餐厅",
            "warnings": [] if items else ["no_dinings_found"],
            "items": items,
        }
        self._report(state, 60, "Dining query completed")
        return {"dinings": payload}

    def _build_original_memory_query(self, request: TripPlanRequest) -> str:
        return " ".join(
            part
            for part in [
                request.destination,
                " ".join(request.preferences or []),
                " ".join(request.hotel_preferences or []),
                request.budget,
                request.special_requirements or "",
            ]
            if part
        )

    def _build_expanded_memory_queries(self, request: TripPlanRequest, query_bundle: Dict[str, Any]) -> List[str]:
        original_query = self._build_original_memory_query(request)
        llm_queries = self._normalize_query_list(query_bundle.get("memory_queries"), request.destination)
        return [query for query in self._dedupe_texts(llm_queries)[:5] if query != original_query]

    def _retrieve_user_memories_three_way(
        self,
        user_id: str,
        *,
        original_query: str,
        expanded_queries: List[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        return self.memory_service.retrieve_user_memories_three_way(
            user_id=user_id,
            query=original_query,
            expanded_queries=expanded_queries,
            limit=limit,
            memory_types=["preference", "trip"],
        )

    def _retrieve_knowledge_memories_three_way(
        self,
        *,
        original_query: str,
        expanded_queries: List[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        return self.memory_service.retrieve_knowledge_memories_three_way(
            query=original_query,
            expanded_queries=expanded_queries,
            limit=limit,
            knowledge_types=["destination", "experience"],
        )

    def _dedupe_ranked_items(self, items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for item in sorted(
            items,
            key=lambda value: (
                value.get("final_rerank_score", 0.0),
                value.get("rerank_score", 0.0),
                value.get("rrf_score", 0.0),
            ),
            reverse=True,
        ):
            key = item.get("metadata_id") or item.get("id") or self._stable_text_key(item)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def _build_attraction_search_terms(
        self,
        request: TripPlanRequest,
        constraints: Dict[str, Any],
    ) -> List[str]:
        terms: List[str] = []
        constraints = constraints or {}
        intents = constraints.get("intents", {}) if isinstance(constraints, dict) else {}
        conflicts = constraints.get("conflicts", []) if isinstance(constraints, dict) else []
        query_bundle = constraints.get("query_bundle", {}) if isinstance(constraints, dict) else {}
        terms.extend(self._normalize_query_list(query_bundle.get("attraction_queries"), request.destination))

        for intent, search_terms in FALLBACK_INTENT_SEARCH_TERMS.items():
            if intents.get(intent):
                terms.extend(search_terms)
        if conflicts:
            terms.extend(["博物馆", "历史文化", "城市地标"])

        if not terms:
            terms.extend(self._travel_intent_search_terms(request.preferences or [], intents) or ["景点"])

        return self._dedupe_texts(terms)[:8]

    def _travel_intent_search_terms(self, preferences: List[str], intents: Dict[str, Any]) -> List[str]:
        terms: List[str] = []
        for preference in preferences:
            text = str(preference).strip()
            if not text:
                continue
            if self._matches_intent(text, "photography"):
                terms.extend(FALLBACK_INTENT_SEARCH_TERMS["photography"])
                continue
            terms.append(text)
        if intents.get("photography"):
            terms.extend(FALLBACK_INTENT_SEARCH_TERMS["photography"])
        return terms

    def _build_dining_search_terms(
        self,
        request: TripPlanRequest,
        constraints: Dict[str, Any],
    ) -> List[str]:
        terms: List[str] = []
        constraints = constraints or {}
        intents = constraints.get("intents", {}) if isinstance(constraints, dict) else {}
        query_bundle = constraints.get("query_bundle", {}) if isinstance(constraints, dict) else {}
        terms.extend(self._normalize_query_list(query_bundle.get("dining_queries"), request.destination))

        if any("美食" in str(item) for item in request.preferences or []):
            terms.extend(["本地菜 餐厅", "特色小吃", "美食"])
        if intents.get("night_view"):
            terms.extend(["夜市 美食", "夜宵 餐厅"])
        if request.special_requirements:
            terms.append(f"{request.special_requirements[:20]} 餐厅")
        terms.extend(["本地特色餐厅", "餐厅"])

        return self._dedupe_texts(terms)[:6]

    def _query_weather_node(self, state: PlannerState) -> Dict[str, Any]:
        request = state["request"]
        forecast = self._amap_weather(request.destination)
        payload = {
            "summary": f"高德地图返回 {len(forecast)} 天天气预报",
            "warnings": [] if forecast else ["no_weather_found"],
            "forecast": self._align_weather_dates(request, forecast),
        }
        self._report(state, 65, "Weather query completed")
        return {"weather": payload}

    def _generate_plan_node(self, state: PlannerState) -> Dict[str, Any]:
        request = state["request"]
        self._report(state, 70, "Information queries completed")
        self._report(state, 80, "Generating itinerary")
        prompt = self._construct_prompt(
            request=request,
            attractions=state.get("attractions", {}),
            hotels=state.get("hotels", {}),
            dinings=state.get("dinings", {}),
            weather=state.get("weather", {}),
            user_memories=state.get("user_memories", []),
            knowledge_memories=state.get("knowledge_memories", []),
            requirement_constraints=state.get("requirement_constraints", {}),
        )
        try:
            response = self.chat_model.bind(response_format={"type": "json_object"}).invoke(
                [
                    SystemMessage(content=self._planner_system_prompt()),
                    HumanMessage(content=prompt),
                ]
            )
            json_plan_str = response.content if isinstance(response.content, str) else json.dumps(response.content)
            plan_data = json.loads(self._extract_json(json_plan_str))
            plan_data = self._normalize_plan_payload(
                plan_data,
                request=request,
                weather=state.get("weather", {}),
            )
            self._report(state, 88, "Validating itinerary")
            validated_plan = TripPlanResponse.model_validate(plan_data)
        except Exception as exc:
            logger.error("LangGraph final planner failed, using fallback trip plan", exc_info=True, extra={"error": str(exc)})
            validated_plan = self._build_candidate_based_plan(
                request=request,
                attractions=state.get("attractions", {}),
                hotels=state.get("hotels", {}),
                dinings=state.get("dinings", {}),
                weather=state.get("weather", {}),
            )

        validated_plan = self._validate_and_filter_plan(
            validated_plan,
            request.destination,
            candidate_context={
                "attractions": state.get("attractions", {}),
                "hotels": state.get("hotels", {}),
                "dinings": state.get("dinings", {}),
            },
        )
        self._enrich_route_metrics(validated_plan)
        if not any(day.attractions for day in validated_plan.days):
            validated_plan = self._build_candidate_based_plan(
                request=request,
                attractions=state.get("attractions", {}),
                hotels=state.get("hotels", {}),
                dinings=state.get("dinings", {}),
                weather=state.get("weather", {}),
            )
            self._enrich_route_metrics(validated_plan)

        self._report(state, 92, "Enriching attraction images")
        self._enrich_images(validated_plan, request.destination)
        self.memory_service.store_user_preference(
            state["user_id"],
            "trip_request",
            {
                "destination": request.destination,
                "preferences": request.preferences,
                "hotel_preferences": request.hotel_preferences,
                "budget": request.budget,
                "special_requirements": request.special_requirements,
                "trip_title": validated_plan.trip_title,
            },
        )
        logger.info("LangGraph trip plan generated", extra={"destination": request.destination, "title": validated_plan.trip_title})
        return {"plan": validated_plan}

    def plan_trip(
        self,
        request: TripPlanRequest,
        user_id: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> TripPlanResponse | None:
        request_id = get_request_id() or f"req_{datetime.now().timestamp()}"
        user_id = user_id or request_id
        try:
            result = self.graph.invoke(
                {
                    "request": request,
                    "user_id": user_id,
                    "request_id": request_id,
                    "progress_callback": progress_callback,
                }
            )
            return result.get("plan") or self._build_fallback_plan(request)
        except Exception as exc:
            logger.error("LangGraph trip workflow failed, using fallback trip plan", exc_info=True, extra={"error": str(exc)})
            return self._build_fallback_plan(request)

    def _amap_text_search(self, *, keywords: str, city: str, limit: int, category: str) -> List[Dict[str, Any]]:
        search_terms = self._build_amap_fallback_queries(keywords=keywords, city=city, category=category)
        for index, search_keywords in enumerate(search_terms):
            mcp_items = self._run_mcp_tool(
                amap_mcp_client.search_pois(
                    keywords=search_keywords,
                    city=city,
                    limit=limit,
                    category=category,
                )
            )
            if isinstance(mcp_items, list) and mcp_items:
                if index > 0:
                    logger.info(
                        "AMap text search fallback hit",
                        extra={
                            "city": city,
                            "category": category,
                            "original_keywords": keywords,
                            "fallback_keywords": search_keywords,
                            "result_count": len(mcp_items),
                        },
                    )
                return mcp_items

            if not settings.AMAP_API_KEY:
                continue
            try:
                response = requests.get(
                    "https://restapi.amap.com/v3/place/text",
                    params={
                        "key": settings.AMAP_API_KEY,
                        "keywords": search_keywords,
                        "city": city,
                        "citylimit": "true",
                        "offset": limit,
                        "page": 1,
                        "extensions": "all",
                    },
                    timeout=20,
                )
                response.raise_for_status()
                data = response.json()
                pois = data.get("pois", []) if data.get("status") == "1" else []
                if pois:
                    if index > 0:
                        logger.info(
                            "AMap REST text search fallback hit",
                            extra={
                                "city": city,
                                "category": category,
                                "original_keywords": keywords,
                                "fallback_keywords": search_keywords,
                                "result_count": len(pois),
                            },
                        )
                    return [self._normalize_poi(poi, category) for poi in pois[:limit]]
            except Exception as exc:
                logger.warning("AMap text search failed", extra={"city": city, "keywords": search_keywords, "error": str(exc)})
        return []

    def _build_amap_fallback_queries(self, *, keywords: str, city: str, category: str) -> List[str]:
        normalized = str(keywords or "").strip()
        if not normalized:
            return [category]

        without_city = normalized.replace(str(city or ""), "").strip()
        pieces = [part for part in re.split(r"[\s，,、]+", without_city) if part]
        queries = [normalized]

        if len(pieces) > 2:
            queries.append(f"{city} {' '.join(pieces[:2])}".strip())
        for piece in pieces[:4]:
            if len(piece) >= 2:
                queries.append(f"{city} {piece}".strip())

        if category == "景点":
            active_intents = [
                intent
                for intent, keywords in TRAVEL_INTENT_KEYWORDS.items()
                if any(keyword in normalized for keyword in keywords)
            ]
            max_terms_per_intent = max(
                (len(FALLBACK_INTENT_SEARCH_TERMS.get(intent, ())) for intent in active_intents),
                default=0,
            )
            for term_index in range(min(max_terms_per_intent, 2)):
                for intent in active_intents:
                    terms = FALLBACK_INTENT_SEARCH_TERMS.get(intent, ())
                    if term_index < len(terms):
                        queries.append(f"{city} {terms[term_index]}".strip())
        elif category == "餐饮":
            if any(word in normalized for word in ("美食", "夜市", "本地", "特色", "小吃")):
                queries.extend([f"{city} 本地美食", f"{city} 特色小吃"])
        elif category == "酒店":
            if any(word in normalized for word in ("地铁", "交通", "便利", "附近")):
                queries.extend([f"{city} 地铁 酒店", f"{city} 交通便利 酒店"])

        generic_by_category = {
            "景点": ["景点", "热门景点"],
            "餐饮": ["美食", "餐厅"],
            "酒店": ["酒店"],
        }
        queries.extend(f"{city} {term}".strip() for term in generic_by_category.get(category, [category]))
        return self._dedupe_texts(queries)[:8]

    def _amap_global_text_search(self, *, keywords: str, limit: int = 3, category: str = "景点") -> List[Dict[str, Any]]:
        mcp_items = self._run_mcp_tool(
            amap_mcp_client.search_pois(
                keywords=keywords,
                city="",
                limit=limit,
                category=category,
                citylimit=False,
            )
        )
        if isinstance(mcp_items, list):
            return mcp_items

        if not settings.AMAP_API_KEY:
            return []
        try:
            response = requests.get(
                "https://restapi.amap.com/v3/place/text",
                params={
                    "key": settings.AMAP_API_KEY,
                    "keywords": keywords,
                    "citylimit": "false",
                    "offset": limit,
                    "page": 1,
                    "extensions": "all",
                },
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            pois = data.get("pois", []) if data.get("status") == "1" else []
            return [self._normalize_poi(poi, category) for poi in pois[:limit]]
        except Exception as exc:
            logger.warning("AMap global text search failed", extra={"keywords": keywords, "error": str(exc)})
            return []

    def _normalize_poi(self, poi: Dict[str, Any], category: str) -> Dict[str, Any]:
        location = self._parse_location(poi.get("location"))
        return {
            "name": poi.get("name", ""),
            "type": poi.get("type") or category,
            "address": poi.get("address") if isinstance(poi.get("address"), str) else "",
            "location": location,
            "rating": (poi.get("biz_ext") or {}).get("rating") or poi.get("rating") or "N/A",
            "price": (poi.get("biz_ext") or {}).get("cost") or "N/A",
            "cityname": poi.get("cityname") if isinstance(poi.get("cityname"), str) else "",
            "adname": poi.get("adname") if isinstance(poi.get("adname"), str) else "",
            "adcode": poi.get("adcode") if isinstance(poi.get("adcode"), str) else "",
        }

    def _parse_location(self, location_text: Any) -> Optional[Dict[str, float]]:
        if not isinstance(location_text, str) or "," not in location_text:
            return None
        try:
            lng, lat = location_text.split(",", 1)
            return {"lat": float(lat), "lng": float(lng)}
        except Exception:
            return None

    def _amap_weather(self, city: str) -> List[Dict[str, Any]]:
        mcp_weather = self._run_mcp_tool(amap_mcp_client.get_weather_forecast(city))
        if isinstance(mcp_weather, list):
            return mcp_weather

        if not settings.AMAP_API_KEY:
            return []
        try:
            response = requests.get(
                "https://restapi.amap.com/v3/weather/weatherInfo",
                params={
                    "key": settings.AMAP_API_KEY,
                    "city": city,
                    "extensions": "all",
                },
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            forecasts = data.get("forecasts", []) if data.get("status") == "1" else []
            if not forecasts:
                return []
            casts = forecasts[0].get("casts", [])
            return [
                {
                    "date": item.get("date", ""),
                    "day_weather": item.get("dayweather", ""),
                    "night_weather": item.get("nightweather", ""),
                    "day_temp": item.get("daytemp", ""),
                    "night_temp": item.get("nighttemp", ""),
                    "day_wind": f"{item.get('daywind', '')}风 {item.get('daypower', '')}级".strip(),
                    "night_wind": f"{item.get('nightwind', '')}风 {item.get('nightpower', '')}级".strip(),
                }
                for item in casts
            ]
        except Exception as exc:
            logger.warning("AMap weather query failed", extra={"city": city, "error": str(exc)})
            return []

    def _run_mcp_tool(self, coroutine) -> Optional[Any]:
        try:
            return self._run_coroutine_sync(coroutine)
        except Exception as exc:
            logger.warning("MCP tool call failed, falling back to local provider", extra={"error": str(exc)})
            return None

    def _run_coroutine_sync(self, coroutine) -> Any:
        try:
            asyncio_running_loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio_running_loop = None

        if asyncio_running_loop and asyncio_running_loop.is_running():
            result: Dict[str, Any] = {}

            def runner() -> None:
                try:
                    result["value"] = asyncio.run(coroutine)
                except Exception as exc:
                    result["error"] = exc

            thread = threading.Thread(target=runner, daemon=True)
            thread.start()
            thread.join()
            if "error" in result:
                raise result["error"]
            return result.get("value")

        return asyncio.run(coroutine)

    def _align_weather_dates(self, request: TripPlanRequest, forecast: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        start = datetime.fromisoformat(request.start_date)
        end = datetime.fromisoformat(request.end_date)
        day_count = max(1, min((end - start).days + 1, 14))
        by_date = {item.get("date"): item for item in forecast}
        aligned = []
        for index in range(day_count):
            date_text = (start + timedelta(days=index)).date().isoformat()
            aligned.append(
                by_date.get(date_text)
                or (forecast[min(index, len(forecast) - 1)] if forecast else None)
                or {
                    "date": date_text,
                    "day_weather": "请以实时天气为准",
                    "night_weather": "请以实时天气为准",
                    "day_temp": "N/A",
                    "night_temp": "N/A",
                    "day_wind": "",
                    "night_wind": "",
                }
            )
            aligned[-1]["date"] = date_text
        return aligned

    def _analyze_requirement_constraints(self, request: TripPlanRequest) -> Dict[str, Any]:
        text = (request.special_requirements or "").strip()
        query_bundle = self._generate_query_bundle(request)
        if not text:
            return {
                "resolved_places": [],
                "conflicts": [],
                "intents": self._extract_requirement_intents(request),
                "query_bundle": query_bundle,
                "notes": [],
            }

        candidate_names = self._extract_requirement_place_candidates(text)
        conflicts: List[Dict[str, Any]] = []
        resolved_places: List[Dict[str, Any]] = []

        for name in candidate_names[:5]:
            pois = self._amap_global_text_search(keywords=name, limit=3, category="景点")
            if not pois:
                continue

            best_poi = self._choose_best_requirement_poi(name, pois)
            location = best_poi.get("location")
            in_destination = self._validate_poi_dict_in_destination(best_poi, request.destination)

            place_info = {
                "query": name,
                "name": best_poi.get("name", name),
                "cityname": best_poi.get("cityname", ""),
                "adname": best_poi.get("adname", ""),
                "adcode": best_poi.get("adcode", ""),
                "address": best_poi.get("address", ""),
                "location": location,
                "in_destination": in_destination,
            }
            resolved_places.append(place_info)
            if not in_destination:
                conflicts.append(place_info)

        if conflicts:
            logger.info(
                "Detected out-of-destination requirements",
                extra={"destination": request.destination, "conflicts": conflicts},
            )

        return {
            "resolved_places": resolved_places,
            "conflicts": conflicts,
            "intents": self._extract_requirement_intents(request),
            "query_bundle": query_bundle,
            "notes": [
                "结构化目的地和日期是硬边界。",
                "勾选偏好与补充需求都是用户需求，应同等重视。",
                "补充需求中的异地景点不得直接加入行程，应把意图转成目标城市内同类替代。",
            ],
        }

    def _generate_query_bundle(self, request: TripPlanRequest) -> Dict[str, Any]:
        fallback = self._fallback_query_bundle(request)
        prompt = f"""
请把旅行需求拆解为多路检索 query。只输出 JSON，不要解释。
目标城市: {request.destination}
出行日期: {request.start_date} 至 {request.end_date}
预算: {request.budget}
勾选偏好: {'、'.join(request.preferences or []) or '无'}
酒店偏好: {'、'.join(request.hotel_preferences or []) or '无'}
补充需求: {request.special_requirements or '无'}

要求:
1. query 必须围绕目标城市，不能推荐其他城市景点。
2. 只改写“需求表达角度”，不要输出具体景点、酒店、餐厅名称。
3. attraction_queries 面向高德景点 POI 搜索，3-5 条，写成“目标城市 + 场景/体验/类型”，例如“桂林 夜景 夜游 游船”，不要写“桂林 两江四湖”。
4. dining_queries 面向高德餐饮 POI 搜索，2-4 条，写成“目标城市 + 菜系/场景/餐饮类型”，不要写具体店名。
5. hotel_queries 面向高德酒店 POI 搜索，1-3 条，写成“目标城市 + 位置/交通/预算/酒店类型”，不要写具体酒店名。
6. memory_queries 面向用户历史偏好/行程记忆检索，3-5 条，也只写偏好和体验，不写具体 POI 名称。
7. must_have 放硬需求，prefer 放偏好，avoid 放排除项，time_constraints 和 location_constraints 放时间/位置约束。
8. 不要把“摄影打卡”固定理解为摄影服务，应理解为旅游拍照、出片、网红打卡、代表性地点。

JSON 字段:
{{
  "memory_queries": [],
  "attraction_queries": [],
  "dining_queries": [],
  "hotel_queries": [],
  "must_have": [],
  "prefer": [],
  "avoid": [],
  "time_constraints": [],
  "location_constraints": []
}}
"""
        try:
            response = self.chat_model.bind(response_format={"type": "json_object"}).invoke(
                [
                    SystemMessage(content="你是旅行检索 query 改写器，只输出合法 JSON。"),
                    HumanMessage(content=prompt),
                ]
            )
            content = response.content if isinstance(response.content, str) else json.dumps(response.content)
            parsed = json.loads(self._extract_json(content))
            if not isinstance(parsed, dict):
                raise ValueError("query bundle is not a JSON object")
            bundle = self._normalize_query_bundle(parsed, request)
            logger.info(
                "Generated multi-query retrieval bundle",
                extra={
                    "destination": request.destination,
                    "attraction_queries": bundle.get("attraction_queries"),
                    "memory_queries": bundle.get("memory_queries"),
                },
            )
            return bundle
        except Exception as exc:
            logger.warning("Failed to generate LLM query bundle, using fallback", extra={"error": str(exc)})
            return fallback

    def _normalize_query_bundle(self, data: Dict[str, Any], request: TripPlanRequest) -> Dict[str, Any]:
        bundle = {key: list(value) for key, value in DEFAULT_QUERY_BUNDLE.items()}
        for key in bundle:
            bundle[key] = self._normalize_query_list(data.get(key), request.destination)
        fallback = self._fallback_query_bundle(request)
        for key in ("memory_queries", "attraction_queries", "dining_queries", "hotel_queries"):
            if not bundle[key]:
                bundle[key] = fallback[key]
        return bundle

    def _fallback_query_bundle(self, request: TripPlanRequest) -> Dict[str, Any]:
        intents = self._extract_requirement_intents(request)
        base = " ".join(
            part
            for part in [
                request.destination,
                " ".join(request.preferences or []),
                request.special_requirements or "",
            ]
            if part
        )
        attraction_queries = [base or f"{request.destination} 景点"]
        active_intents = [
            intent
            for intent in FALLBACK_INTENT_SEARCH_TERMS
            if intents.get(intent)
        ]
        max_terms_per_intent = max(
            (len(FALLBACK_INTENT_SEARCH_TERMS[intent]) for intent in active_intents),
            default=0,
        )
        for term_index in range(max_terms_per_intent):
            for intent in active_intents:
                search_terms = FALLBACK_INTENT_SEARCH_TERMS[intent]
                if term_index < len(search_terms):
                    attraction_queries.append(f"{request.destination} {search_terms[term_index]}")
        dining_queries = [f"{request.destination} 本地特色餐厅", f"{request.destination} 美食"]
        if intents.get("night_view"):
            dining_queries.append(f"{request.destination} 夜市 美食")
        hotel_query = request.hotel_preferences[0] if request.hotel_preferences else "酒店"
        hotel_queries = [f"{request.destination} {hotel_query if '酒店' in hotel_query else hotel_query + ' 酒店'}"]
        if intents.get("transit_hotel"):
            hotel_queries.insert(0, f"{request.destination} 地铁 酒店")
        memory_queries = [base or f"{request.destination} 旅行偏好", *attraction_queries[:3]]
        return {
            "memory_queries": self._dedupe_texts(memory_queries)[:5],
            "attraction_queries": self._dedupe_texts(attraction_queries)[:10],
            "dining_queries": self._dedupe_texts(dining_queries)[:4],
            "hotel_queries": self._dedupe_texts(hotel_queries)[:3],
            "must_have": [],
            "prefer": list(request.preferences or []),
            "avoid": [],
            "time_constraints": [],
            "location_constraints": [],
        }

    def _extract_requirement_intents(self, request: TripPlanRequest) -> Dict[str, bool]:
        text = " ".join(
            [
                request.special_requirements or "",
                " ".join(request.preferences or []),
                " ".join(request.hotel_preferences or []),
            ]
        )
        return {intent: self._matches_intent(text, intent) for intent in TRAVEL_INTENT_KEYWORDS}

    def _matches_intent(self, text: str, intent: str) -> bool:
        return any(keyword in text for keyword in TRAVEL_INTENT_KEYWORDS.get(intent, ()))

    def _extract_requirement_place_candidates(self, text: str) -> List[str]:
        stop_words = {
            "想去",
            "希望",
            "尽量",
            "不要",
            "不想",
            "酒店",
            "靠近",
            "地铁",
            "晚上",
            "白天",
            "夜景",
            "适合",
            "拍照",
            "打卡",
            "太多路",
            "少走路",
        }
        candidates: List[str] = []
        for chunk in re.split(r"[，,。.!！?？；;\s、和与及]+", text):
            normalized = chunk.strip(" ，,。.!！?？；;：:")
            if not normalized or normalized in stop_words:
                continue
            normalized = re.sub(r"^(想去|希望去|打算去|一定要去|必须去)", "", normalized)
            normalized = re.sub(r"(附近|周边|里面|内|景区|博物馆)$", lambda m: m.group(0), normalized)
            if 2 <= len(normalized) <= 12 and not any(word in normalized for word in stop_words):
                candidates.append(normalized)

        deduped: List[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _choose_best_requirement_poi(self, query: str, pois: List[Dict[str, Any]]) -> Dict[str, Any]:
        def score(poi: Dict[str, Any]) -> tuple[int, float]:
            name = str(poi.get("name", ""))
            exact = 2 if name == query else 1 if query in name or name in query else 0
            try:
                rating = float(poi.get("rating") or 0)
            except Exception:
                rating = 0.0
            return exact, rating

        return sorted(pois, key=score, reverse=True)[0]

    def _dedupe_pois(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for item in items:
            name = str(item.get("name") or "").strip()
            address = str(item.get("address") or "").strip()
            location = item.get("location") or {}
            loc_key = ""
            if isinstance(location, dict):
                loc_key = f"{location.get('lat')}:{location.get('lng')}"
            key = (name, address, loc_key)
            if not name or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _dedupe_texts(self, texts: List[Any]) -> List[str]:
        deduped: List[str] = []
        for text in texts:
            clean = str(text or "").strip()
            if clean and clean not in deduped:
                deduped.append(clean)
        return deduped

    def _normalize_query_list(self, value: Any, destination: str) -> List[str]:
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, list):
            raw_items = value
        else:
            raw_items = []

        queries: List[str] = []
        for item in raw_items:
            text = str(item or "").strip()
            if not text:
                continue
            if destination and destination not in text:
                text = f"{destination} {text}"
            queries.append(text)
        return self._dedupe_texts(queries)

    def _stable_text_key(self, item: Dict[str, Any]) -> str:
        return "|".join(
            str(item.get(key, "") or "")
            for key in ("type", "preference_type", "text_representation", "content", "destination")
        )

    def _rank_pois_for_request(
        self,
        items: List[Dict[str, Any]],
        *,
        request: TripPlanRequest,
        constraints: Dict[str, Any],
        poi_kind: str,
    ) -> List[Dict[str, Any]]:
        intents = constraints.get("intents", {}) if isinstance(constraints, dict) else {}
        conflicts = constraints.get("conflicts", []) if isinstance(constraints, dict) else []
        items = self._filter_pois_for_request(items, request=request, constraints=constraints, poi_kind=poi_kind)
        rerank_query = self._build_poi_rerank_query(request, constraints, poi_kind)
        documents = [self._poi_to_rerank_document(item, poi_kind) for item in items]
        rerank_results = rerank_service.rerank(
            query=rerank_query,
            documents=documents,
            top_n=min(settings.RERANK_TOP_N, len(documents)),
        )
        rerank_scores = {
            result["index"]: result["relevance_score"]
            for result in rerank_results
            if 0 <= result.get("index", -1) < len(items)
        }

        for index, item in enumerate(items):
            semantic_score = rerank_scores.get(index, 0.0)
            item["semantic_rerank_score"] = semantic_score
            item["final_rerank_score"] = semantic_score

        ranked = sorted(items, key=lambda item: item.get("final_rerank_score", 0.0), reverse=True)
        logger.info(
            "Ranked POIs by user requirements",
            extra={
                "destination": request.destination,
                "poi_kind": poi_kind,
                "intents": intents,
                "rerank_model": settings.RERANK_MODEL if settings.RERANK_ENABLED else "disabled",
                "ranking_strategy": "semantic_rerank_only_with_fact_filters",
                "top_items": [
                    {
                        "name": item.get("name"),
                        "semantic": item.get("semantic_rerank_score"),
                        "final": item.get("final_rerank_score"),
                    }
                    for item in ranked[:5]
                ],
            },
        )
        return ranked

    def _filter_pois_for_request(
        self,
        items: List[Dict[str, Any]],
        *,
        request: TripPlanRequest,
        constraints: Dict[str, Any],
        poi_kind: str,
    ) -> List[Dict[str, Any]]:
        constraints = constraints or {}
        filtered: List[Dict[str, Any]] = []
        rejected_count = 0
        for item in items:
            if poi_kind == "attraction" and not self._looks_like_visit_poi(item):
                rejected_count += 1
                continue
            filtered.append(item)
        if rejected_count:
            logger.info(
                "Filtered POIs by factual type constraints",
                extra={
                    "destination": request.destination,
                    "poi_kind": poi_kind,
                    "rejected_count": rejected_count,
                    "kept_count": len(filtered),
                },
            )
        return filtered

    def _looks_like_visit_poi(self, item: Dict[str, Any]) -> bool:
        poi_type = str(item.get("type") or "")
        if any(category in poi_type for category in NON_VISIT_POI_TYPE_CATEGORIES):
            return False
        return any(category in poi_type for category in VISIT_POI_TYPE_CATEGORIES)

    def _build_poi_rerank_query(
        self,
        request: TripPlanRequest,
        constraints: Dict[str, Any],
        poi_kind: str,
    ) -> str:
        intents = constraints.get("intents", {}) if isinstance(constraints, dict) else {}
        conflicts = constraints.get("conflicts", []) if isinstance(constraints, dict) else []
        query_bundle = constraints.get("query_bundle", {}) if isinstance(constraints, dict) else {}
        intent_text = []
        for intent, description in TRAVEL_INTENT_RERANK_TEXT.items():
            if intents.get(intent):
                intent_text.append(description)
        if conflicts:
            conflict_names = "、".join(str(item.get("query") or item.get("name")) for item in conflicts)
            intent_text.append(f"用户提到的{conflict_names}不在{request.destination}，需要在{request.destination}找同类型文化、博物馆、地标替代")

        base = [
            f"目的地: {request.destination}",
            f"候选类型: {self._poi_kind_label(poi_kind)}",
            f"勾选偏好: {'、'.join(request.preferences or []) or '无'}",
            f"酒店偏好: {'、'.join(request.hotel_preferences or []) or '无'}",
            f"补充需求: {request.special_requirements or '无'}",
            f"有效意图: {'；'.join(intent_text) or '无'}",
            f"硬需求: {'、'.join(query_bundle.get('must_have') or []) or '无'}",
            f"偏好需求: {'、'.join(query_bundle.get('prefer') or []) or '无'}",
            f"排除项: {'、'.join(query_bundle.get('avoid') or []) or '无'}",
            f"时间约束: {'、'.join(query_bundle.get('time_constraints') or []) or '无'}",
            f"位置约束: {'、'.join(query_bundle.get('location_constraints') or []) or '无'}",
        ]
        return "\n".join(base)

    def _poi_kind_label(self, poi_kind: str) -> str:
        return {
            "hotel": "酒店",
            "dining": "餐厅",
            "attraction": "景点",
        }.get(poi_kind, poi_kind)

    def _poi_to_rerank_document(self, item: Dict[str, Any], poi_kind: str) -> str:
        fields = [
            f"名称: {item.get('name') or ''}",
            f"类型: {item.get('type') or poi_kind}",
            f"地址: {item.get('address') or ''}",
            f"城市: {item.get('cityname') or ''}",
            f"区域: {item.get('adname') or ''}",
            f"评分: {item.get('rating') or 'N/A'}",
            f"旅行候选类型: {self._poi_kind_label(poi_kind)}",
        ]
        return "\n".join(fields)

    def _planner_system_prompt(self) -> str:
        return """
你是一个严谨的旅行规划专家。你必须只输出合法 JSON，不要输出 Markdown 或解释。
你会收到高德地图景点、酒店、天气候选信息，请优先使用候选中的真实名称、地址和坐标。
不要推荐目标城市之外的景点。不要在多天中重复同一个景点或餐厅。
预算 total 必须等于各项费用之和。图片 URL 只允许放在 attractions.image_urls 字段中。
"""

    def _compact_location(self, item: Dict[str, Any]) -> Dict[str, float]:
        location = item.get("location") or {}
        return {
            "lat": float(location.get("lat") or item.get("lat") or 0.0),
            "lng": float(location.get("lng") or item.get("lng") or 0.0),
        }

    def _compact_poi_for_prompt(self, item: Dict[str, Any], *, include_price: bool = False) -> Dict[str, Any]:
        compact = {
            "name": item.get("name") or "",
            "type": item.get("type") or "",
            "address": item.get("address") or "",
            "location": self._compact_location(item),
            "rating": item.get("rating") or "",
        }
        if include_price:
            compact["price"] = item.get("price") or item.get("cost_per_person") or ""
        ticket_price = item.get("ticket_price")
        if ticket_price:
            compact["ticket_price"] = ticket_price
        return compact

    def _compact_weather_for_prompt(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "date": item.get("date") or "",
            "day_weather": item.get("day_weather") or item.get("weather") or "",
            "night_weather": item.get("night_weather") or "",
            "day_temp": item.get("day_temp") or "",
            "night_temp": item.get("night_temp") or "",
            "day_wind": item.get("day_wind") or "",
            "night_wind": item.get("night_wind") or "",
        }

    def _construct_prompt(
        self,
        *,
        request: TripPlanRequest,
        attractions: Dict[str, Any],
        hotels: Dict[str, Any],
        dinings: Dict[str, Any],
        weather: Dict[str, Any],
        user_memories: List[Dict[str, Any]],
        knowledge_memories: List[Dict[str, Any]],
        requirement_constraints: Dict[str, Any],
    ) -> str:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        duration = (end_date - start_date).days + 1
        attraction_items = [
            self._compact_poi_for_prompt(item)
            for item in attractions.get("items", [])[: min(8, max(duration * 3, 6))]
        ]
        hotel_items = [
            self._compact_poi_for_prompt(item, include_price=True)
            for item in hotels.get("items", [])[:3]
        ]
        dining_items = [
            self._compact_poi_for_prompt(item, include_price=True)
            for item in dinings.get("items", [])[: min(8, max(duration * 2, 4))]
        ]
        weather_items = [
            self._compact_weather_for_prompt(item)
            for item in weather.get("forecast", [])[:duration]
        ]
        memory_text = [
            item.get("text_representation", "")[:80]
            for item in (user_memories or [])[:2]
            if isinstance(item, dict)
        ]
        constraint_text = self._format_requirement_constraints(request, requirement_constraints)
        return f"""
请为我创建一个前往 {request.destination} 的旅行计划。

基本信息:
- 旅行天数: {duration} 天，从 {request.start_date} 到 {request.end_date}
- 预算水平: {request.budget}
- 个人偏好: {', '.join(request.preferences) if request.preferences else '无'}
- 酒店偏好: {', '.join(request.hotel_preferences) if request.hotel_preferences else '无'}
- 补充需求: {request.special_requirements or '无'}
- 可参考历史记忆: {'; '.join(memory_text) if memory_text else '无'}

补充需求解析与硬约束:
{constraint_text}

结构化景点候选:
{json.dumps(attraction_items, ensure_ascii=False)}

结构化酒店候选:
{json.dumps(hotel_items, ensure_ascii=False)}

结构化餐厅候选:
{json.dumps(dining_items, ensure_ascii=False)}

结构化天气信息:
{json.dumps(weather_items, ensure_ascii=False)}

必须返回合法 JSON，顶层字段固定为:
trip_title, total_budget, hotels, days。
total_budget 和 days[].budget 固定字段:
transport_cost, dining_cost, hotel_cost, attraction_ticket_cost, total。
hotels[]/recommended_hotel 字段:
name, address, location{{lat,lng}}, price, rating, distance_to_main_attraction_km。
days[] 字段:
day, theme, weather, recommended_hotel, attractions, dinings, budget。
attractions[] 字段:
name, type, rating, suggested_duration_hours, description, address, location{{lat,lng}}, image_urls, ticket_price。
dinings[] 字段:
name, address, location{{lat,lng}}, cost_per_person, rating。

要求:
1. 每天 2 到 3 个景点，优先使用候选景点的真实名称、地址、location。
2. 每天 1 到 2 家餐厅，优先使用候选餐厅的真实名称、地址、location。
3. 结构化目的地 {request.destination} 是硬约束，补充需求是软约束；如果补充需求中的景点不属于 {request.destination}，不要加入行程，应改为推荐 {request.destination} 内同类型替代。
4. 补充需求中出现的明确景点、节奏、交通、酒店位置、禁忌或偏好，在不违反目的地硬约束的前提下优先满足。
5. 描述控制在 50 字以内，避免长篇解释。
6. 只输出 JSON，不要输出 Markdown。
"""

    def _format_requirement_constraints(self, request: TripPlanRequest, constraints: Dict[str, Any]) -> str:
        if not request.special_requirements:
            return f"- 未填写补充需求。\n- 目的地 {request.destination} 仍然是唯一城市范围。"

        constraints = constraints or {}
        resolved_places = constraints.get("resolved_places") or []
        conflicts = constraints.get("conflicts") or []
        intents = constraints.get("intents") or {}
        intent_labels = {
            "photography": "摄影/出片",
            "nature": "自然风光",
            "family": "亲子友好",
            "low_walking": "少走路/轻松节奏",
            "transit_hotel": "酒店靠近地铁/交通便利",
            "water": "玩水/亲水体验",
            "night_view": "夜景/夜游",
        }
        active_intents = [label for key, label in intent_labels.items() if intents.get(key)]
        lines = [
            f"- 目的地 {request.destination} 是硬约束，行程景点、酒店和餐厅都必须在该城市范围内。",
            "- 勾选偏好和补充需求都是用户需求，应同等重视；只有与目的地事实冲突的地点不能直接安排。",
        ]
        if active_intents:
            lines.append(f"- 已识别有效需求意图: {'、'.join(active_intents)}。")
        query_bundle = constraints.get("query_bundle") or {}
        structured_requirements = []
        for label, key in (
            ("硬需求", "must_have"),
            ("偏好", "prefer"),
            ("排除项", "avoid"),
            ("时间约束", "time_constraints"),
            ("位置约束", "location_constraints"),
        ):
            values = query_bundle.get(key) or []
            if values:
                structured_requirements.append(f"{label}: {'、'.join(values)}")
        if structured_requirements:
            lines.append(f"- LLM 已拆解补充需求: {'; '.join(structured_requirements)}。")
        if resolved_places:
            resolved_text = []
            for place in resolved_places[:5]:
                name = place.get("name") or place.get("query") or "未知地点"
                city = place.get("cityname") or place.get("adname") or "未知城市"
                status = "目标城市内" if place.get("in_destination") else "目标城市外"
                resolved_text.append(f"{name}({city}, {status})")
            lines.append(f"- 已解析补充需求地点: {'; '.join(resolved_text)}。")
        if conflicts:
            conflict_text = []
            for place in conflicts[:5]:
                query = place.get("query") or place.get("name") or "未知地点"
                name = place.get("name") or query
                city = place.get("cityname") or place.get("adname") or "目标城市外"
                conflict_text.append(f"用户提到“{query}”，识别为“{name}”，位置在{city}，不属于{request.destination}")
            lines.append(f"- 异地冲突: {'; '.join(conflict_text)}。这些地点不得进入行程。")
            lines.append(f"- 不要忽略异地景点背后的意图。请在 {request.destination} 内选择同类型替代，例如同为历史文化、博物馆、城市地标或亲子友好景点。")
        if intents.get("low_walking"):
            lines.append("- 用户明确不想走太多路；候选 POI 已经过语义重排，请优先从排序靠前的低步行强度候选中组织行程。")
        if intents.get("transit_hotel"):
            lines.append("- 用户要求酒店靠近地铁；候选酒店已按该需求语义重排，请优先使用排序靠前的酒店。")
        if intents.get("water"):
            lines.append("- 用户想玩水；候选景点已按漂流、竹筏、游船、亲水体验等需求语义重排，请优先使用排序靠前的相关景点。")
        if intents.get("night_view"):
            lines.append("- 用户想看夜景；候选景点已按夜游、灯光、夜市等需求语义重排，请在行程中保留夜间安排。")
        if not conflicts:
            lines.append("- 未发现补充需求中的异地景点冲突。")
        return "\n".join(lines)

    def _extract_json(self, text: str) -> str:
        if "```json" in text:
            return text.split("```json", 1)[1].split("```", 1)[0].strip()
        if "```" in text:
            return text.split("```", 1)[1].split("```", 1)[0].strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else text

    def _normalize_plan_payload(
        self,
        plan_data: Dict[str, Any],
        *,
        request: TripPlanRequest,
        weather: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(plan_data, dict):
            return plan_data
        forecast = list((weather or {}).get("forecast") or [])
        for index, day in enumerate(plan_data.get("days") or []):
            if not isinstance(day, dict):
                continue
            if not isinstance(day.get("day"), int):
                day["day"] = index + 1
            weather_value = day.get("weather")
            if not isinstance(weather_value, dict):
                fallback_weather = forecast[index] if index < len(forecast) else {}
                try:
                    date_text = (datetime.fromisoformat(request.start_date) + timedelta(days=index)).date().isoformat()
                except Exception:
                    date_text = request.start_date
                day["weather"] = {
                    "date": fallback_weather.get("date") or date_text,
                    "day_weather": str(weather_value or fallback_weather.get("day_weather") or "以实时天气为准"),
                    "night_weather": str(fallback_weather.get("night_weather") or weather_value or "以实时天气为准"),
                    "day_temp": str(fallback_weather.get("day_temp") or "N/A"),
                    "night_temp": str(fallback_weather.get("night_temp") or "N/A"),
                    "day_wind": fallback_weather.get("day_wind"),
                    "night_wind": fallback_weather.get("night_wind"),
                }
        return plan_data

    def _enrich_images(self, plan: TripPlanResponse, destination: str) -> None:
        attractions = [attraction for day in plan.days for attraction in day.attractions]
        if not attractions:
            return
        try:
            self.unsplash_service.enrich_attractions(
                attractions=attractions,
                destination=destination,
                use_fallback=True,
                use_cache=True,
            )
        except Exception as exc:
            logger.error("Attraction image enrichment failed", extra={"error": str(exc)})
            for attraction in attractions:
                attraction.image_urls = []

    def _build_candidate_based_plan(
        self,
        *,
        request: TripPlanRequest,
        attractions: Dict[str, Any],
        hotels: Dict[str, Any],
        dinings: Dict[str, Any],
        weather: Dict[str, Any],
    ) -> TripPlanResponse:
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)
        day_count = max(1, min((end_date - start_date).days + 1, 7))
        attraction_items = list(attractions.get("items") or [])
        hotel_items = list(hotels.get("items") or [])
        dining_items = list(dinings.get("items") or [])
        weather_items = list(weather.get("forecast") or [])

        hotel_models = [self._hotel_from_candidate(item) for item in hotel_items[:3]]
        if not hotel_models:
            hotel_models = [
                Hotel(
                    name=f"{request.destination}市中心酒店",
                    address=f"{request.destination}市中心",
                    price=request.budget,
                    rating="N/A",
                )
            ]

        days: List[DailyPlan] = []
        per_day = 2 if len(attraction_items) >= day_count * 2 else 1
        for index in range(day_count):
            date_text = (start_date + timedelta(days=index)).date().isoformat()
            selected = attraction_items[index * per_day : (index + 1) * per_day]
            if not selected and attraction_items:
                selected = [attraction_items[index % len(attraction_items)]]
            attraction_models = [
                self._attraction_from_candidate(item, request.destination, index)
                for item in selected
            ]
            if not attraction_models:
                attraction_models = [
                    Attraction(
                        name=f"{request.destination}城市核心游览区",
                        type="城市游览",
                        rating="N/A",
                        suggested_duration_hours=3,
                        description="结合当地交通和实时开放信息安排游览。",
                        address=f"{request.destination}市区",
                        ticket_price="N/A",
                    )
                ]

            recommended_hotel = hotel_models[index % len(hotel_models)]
            weather_data = self._weather_from_candidate(weather_items, index, date_text)
            selected_dinings = dining_items[index * 2 : (index + 1) * 2]
            dining_models = [self._dining_from_candidate(item) for item in selected_dinings]
            if not dining_models:
                dining_models = [
                    Dining(
                        name=f"{request.destination}本地特色餐厅",
                        address=attraction_models[0].address or f"{request.destination}市区",
                        location=attraction_models[0].location,
                        cost_per_person="90" if request.budget in {"中等", "medium"} else "60",
                        rating="N/A",
                    )
                ]
            dining_cost = 180.0 if request.budget in {"中等", "medium"} else 120.0
            hotel_cost = self._parse_price(recommended_hotel.price) or (300.0 if request.budget in {"中等", "medium"} else 180.0)
            ticket_cost = sum(self._parse_price(item.ticket_price) or 0.0 for item in attraction_models)
            daily_budget = DailyBudget(
                transport_cost=80.0,
                dining_cost=dining_cost,
                hotel_cost=hotel_cost,
                attraction_ticket_cost=ticket_cost,
                total=80.0 + dining_cost + hotel_cost + ticket_cost,
            )
            days.append(
                DailyPlan(
                    day=index + 1,
                    theme=f"{request.destination}精选体验第{index + 1}天",
                    weather=weather_data,
                    recommended_hotel=recommended_hotel,
                    attractions=attraction_models,
                    dinings=dining_models,
                    budget=daily_budget,
                )
            )

        total_budget = BudgetBreakdown(
            transport_cost=sum(day.budget.transport_cost for day in days),
            dining_cost=sum(day.budget.dining_cost for day in days),
            hotel_cost=sum(day.budget.hotel_cost for day in days),
            attraction_ticket_cost=sum(day.budget.attraction_ticket_cost for day in days),
        )
        total_budget.total = (
            total_budget.transport_cost
            + total_budget.dining_cost
            + total_budget.hotel_cost
            + total_budget.attraction_ticket_cost
        )
        return TripPlanResponse(
            trip_title=f"{request.destination}{day_count}日{request.preferences[0] if request.preferences else '精选'}行程",
            total_budget=total_budget,
            hotels=hotel_models,
            days=days,
        )

    def _attraction_from_candidate(self, item: Dict[str, Any], destination: str, index: int) -> Attraction:
        location = item.get("location")
        return Attraction(
            name=item.get("name") or f"{destination}精选景点{index + 1}",
            type=item.get("type") or "景点",
            rating=item.get("rating") or "N/A",
            suggested_duration_hours=2.0,
            description="根据偏好和地理位置安排游览，建议提前确认开放时间和交通。",
            address=item.get("address") or f"{destination}市区",
            location=Location(**location) if isinstance(location, dict) else None,
            cityname=item.get("cityname") or "",
            adname=item.get("adname") or "",
            adcode=item.get("adcode") or "",
            ticket_price="N/A",
        )

    def _hotel_from_candidate(self, item: Dict[str, Any]) -> Hotel:
        location = item.get("location")
        price = item.get("price")
        if not price or price == "N/A":
            price = "价格以平台为准"
        return Hotel(
            name=item.get("name") or "推荐酒店",
            address=item.get("address") or "",
            location=Location(**location) if isinstance(location, dict) else None,
            cityname=item.get("cityname") or "",
            adname=item.get("adname") or "",
            adcode=item.get("adcode") or "",
            price=price,
            rating=item.get("rating") or "N/A",
        )

    def _dining_from_candidate(self, item: Dict[str, Any]) -> Dining:
        location = item.get("location")
        return Dining(
            name=item.get("name") or "推荐餐厅",
            address=item.get("address") or "",
            location=Location(**location) if isinstance(location, dict) else None,
            cityname=item.get("cityname") or "",
            adname=item.get("adname") or "",
            adcode=item.get("adcode") or "",
            cost_per_person=item.get("price") or item.get("cost_per_person") or "N/A",
            rating=item.get("rating") or "N/A",
        )

    def _weather_from_candidate(self, weather_items: List[Dict[str, Any]], index: int, date_text: str) -> Weather:
        item = weather_items[index] if index < len(weather_items) else {}
        return Weather(
            date=date_text,
            day_weather=item.get("day_weather") or "请以实时天气为准",
            night_weather=item.get("night_weather") or "请以实时天气为准",
            day_temp=str(item.get("day_temp") or "N/A"),
            night_temp=str(item.get("night_temp") or "N/A"),
            day_wind=item.get("day_wind"),
            night_wind=item.get("night_wind"),
        )

    def _parse_price(self, value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None
        match = re.search(r"\d+(?:\.\d+)?", value)
        return float(match.group(0)) if match else None

    def _normalize_city_name(self, value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"(市|地区|盟|自治州|特别行政区)$", "", value.strip())

    def _admin_matches_destination(self, cityname: Any, adname: Any, destination: str) -> Optional[bool]:
        destination_norm = self._normalize_city_name(destination)
        city_norm = self._normalize_city_name(cityname)
        ad_norm = self._normalize_city_name(adname)
        if city_norm:
            return destination_norm == city_norm
        if ad_norm and destination_norm == ad_norm:
            return True
        return None

    def _reverse_geocode_admin(self, lat: float, lng: float) -> Dict[str, str]:
        cache_key = f"{lat:.6f},{lng:.6f}"
        if cache_key in self._regeo_cache:
            return self._regeo_cache[cache_key]
        if not settings.AMAP_API_KEY:
            return {}
        try:
            response = requests.get(
                "https://restapi.amap.com/v3/geocode/regeo",
                params={
                    "key": settings.AMAP_API_KEY,
                    "location": f"{lng},{lat}",
                    "extensions": "base",
                    "radius": 1000,
                },
                timeout=8,
            )
            response.raise_for_status()
            data = response.json()
            address = data.get("regeocode", {}).get("addressComponent", {}) if data.get("status") == "1" else {}
            admin = {
                "cityname": address.get("city") if isinstance(address.get("city"), str) else "",
                "adname": address.get("district") if isinstance(address.get("district"), str) else "",
                "adcode": address.get("adcode") if isinstance(address.get("adcode"), str) else "",
            }
            self._regeo_cache[cache_key] = admin
            return admin
        except Exception as exc:
            logger.warning("AMap reverse geocode failed", extra={"lat": lat, "lng": lng, "error": str(exc)})
            return {}

    def _validate_poi_dict_in_destination(self, poi: Dict[str, Any], destination: str) -> bool:
        admin_match = self._admin_matches_destination(poi.get("cityname"), poi.get("adname"), destination)
        if admin_match is not None:
            return admin_match

        location = poi.get("location")
        if not isinstance(location, dict):
            return True

        try:
            lat = float(location.get("lat"))
            lng = float(location.get("lng"))
        except (TypeError, ValueError):
            return True

        admin = self._reverse_geocode_admin(lat, lng)
        admin_match = self._admin_matches_destination(admin.get("cityname"), admin.get("adname"), destination)
        if admin_match is not None:
            poi.update({key: value for key, value in admin.items() if value})
            return admin_match

        return self._validate_location_in_city(lat, lng, destination)

    def _amap_route_plan(self, origin: Location, destination: Location, mode: str = "walking") -> Dict[str, Any]:
        mcp_result = self._run_mcp_tool(
            amap_mcp_client.get_route_plan(
                origin_lng=float(origin.lng),
                origin_lat=float(origin.lat),
                destination_lng=float(destination.lng),
                destination_lat=float(destination.lat),
                mode=mode,
            )
        )
        if isinstance(mcp_result, dict) and mcp_result:
            return mcp_result

        if not settings.AMAP_API_KEY:
            return {}

        endpoint = (
            "https://restapi.amap.com/v3/direction/walking"
            if mode == "walking"
            else "https://restapi.amap.com/v3/direction/driving"
        )
        try:
            response = requests.get(
                endpoint,
                params={
                    "key": settings.AMAP_API_KEY,
                    "origin": f"{origin.lng},{origin.lat}",
                    "destination": f"{destination.lng},{destination.lat}",
                },
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            route = data.get("route", {}) if data.get("status") == "1" else {}
            paths = route.get("paths") or []
            if not paths:
                return {}
            path = paths[0]
            return {
                "mode": mode,
                "distance_km": round(float(path.get("distance") or 0) / 1000, 2),
                "duration_minutes": round(float(path.get("duration") or 0) / 60, 1),
            }
        except Exception as exc:
            logger.warning("AMap route planning failed", extra={"mode": mode, "error": str(exc)})
            return {}

    def _route_between(self, origin: Location, destination: Location) -> Dict[str, Any]:
        route = self._amap_route_plan(origin, destination, mode="walking")
        if route and float(route.get("distance_km") or 0) <= 2.5:
            return route
        return self._amap_route_plan(origin, destination, mode="driving")

    def _set_route_metrics(self, place: Any, route: Dict[str, Any]) -> None:
        if not route:
            return
        if hasattr(place, "distance_from_previous_km"):
            place.distance_from_previous_km = route.get("distance_km")
        if hasattr(place, "duration_from_previous_minutes"):
            place.duration_from_previous_minutes = route.get("duration_minutes")
        if hasattr(place, "route_mode"):
            place.route_mode = route.get("mode")

    def _enrich_route_metrics(self, plan: TripPlanResponse) -> None:
        for day in plan.days:
            route_places: List[Any] = []
            if day.recommended_hotel and day.recommended_hotel.location:
                route_places.append(day.recommended_hotel)
            route_places.extend([item for item in day.attractions if item.location])
            route_places.extend([item for item in day.dinings if item.location])

            previous = None
            for place in route_places:
                if previous and previous.location and place.location:
                    route = self._route_between(previous.location, place.location)
                    self._set_route_metrics(place, route)
                previous = place

    def _validate_location_in_city(self, lat: float, lng: float, city: str) -> bool:
        bounds = city_support_service.get_bounds(city) or CITY_BOUNDS.get(city)
        if not bounds:
            logger.warning("City has no configured bounds", extra={"city": city})
            return False
        return bounds["lat_min"] <= lat <= bounds["lat_max"] and bounds["lng_min"] <= lng <= bounds["lng_max"]

    def _validate_place_in_destination(self, place: Any, destination: str) -> bool:
        admin_match = self._admin_matches_destination(
            getattr(place, "cityname", ""),
            getattr(place, "adname", ""),
            destination,
        )
        if admin_match is not None:
            return admin_match

        location = getattr(place, "location", None)
        if not location:
            return True

        lat = float(location.lat)
        lng = float(location.lng)
        admin = self._reverse_geocode_admin(lat, lng)
        admin_match = self._admin_matches_destination(admin.get("cityname"), admin.get("adname"), destination)
        if admin_match is not None:
            for field in ("cityname", "adname", "adcode"):
                if hasattr(place, field) and admin.get(field):
                    setattr(place, field, admin[field])
            return admin_match

        return self._validate_location_in_city(lat, lng, destination)

    def _calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        radius_km = 6371
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
        )
        return radius_km * 2 * math.asin(math.sqrt(a))

    def _candidate_lookup_key(self, name: Any, location: Any = None) -> str:
        name_part = str(name or "").strip().lower()
        if isinstance(location, dict):
            return f"{name_part}|{location.get('lat')}|{location.get('lng')}"
        if location and hasattr(location, "lat") and hasattr(location, "lng"):
            return f"{name_part}|{location.lat}|{location.lng}"
        return name_part

    def _build_candidate_admin_index(self, candidate_context: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not candidate_context:
            return {}
        rows: List[Dict[str, Any]] = []
        for section in ("attractions", "hotels", "dinings"):
            payload = candidate_context.get(section) or {}
            if isinstance(payload, dict):
                rows.extend([item for item in payload.get("items") or [] if isinstance(item, dict)])

        index: Dict[str, Dict[str, Any]] = {}
        for item in rows:
            keys = {
                self._candidate_lookup_key(item.get("name")),
                self._candidate_lookup_key(item.get("name"), item.get("location")),
            }
            for key in keys:
                if key:
                    index[key] = item
        return index

    def _candidate_items_by_section(
        self,
        candidate_context: Optional[Dict[str, Any]],
        section: str,
    ) -> List[Dict[str, Any]]:
        if not candidate_context:
            return []
        payload = candidate_context.get(section) or {}
        if not isinstance(payload, dict):
            return []
        return [item for item in payload.get("items") or [] if isinstance(item, dict)]

    def _place_matches_candidate(self, place: Any, candidates: List[Dict[str, Any]]) -> bool:
        if not place:
            return False
        keys = {
            self._candidate_lookup_key(getattr(place, "name", "")),
            self._candidate_lookup_key(getattr(place, "name", ""), getattr(place, "location", None)),
        }
        candidate_keys = set()
        for item in candidates:
            candidate_keys.add(self._candidate_lookup_key(item.get("name")))
            candidate_keys.add(self._candidate_lookup_key(item.get("name"), item.get("location")))
        return any(key and key in candidate_keys for key in keys)

    def _needs_candidate_replacement(self, place: Any, candidates: List[Dict[str, Any]], destination: str) -> bool:
        if not place:
            return True
        if not self._place_matches_candidate(place, candidates):
            return True
        if getattr(place, "location", None) is None:
            return True
        return not self._validate_place_in_destination(place, destination)

    def _next_candidate(
        self,
        candidates: List[Dict[str, Any]],
        used_names: set[str],
        *,
        destination: str,
        poi_kind: str,
    ) -> Optional[Dict[str, Any]]:
        for item in candidates:
            name = str(item.get("name") or "")
            if not name or name in used_names:
                continue
            if poi_kind == "attraction" and not self._looks_like_visit_poi(item):
                continue
            if not self._validate_poi_dict_in_destination(item, destination):
                continue
            if not item.get("location"):
                continue
            used_names.add(name)
            return item
        return None

    def _replace_place_from_candidate(self, item: Dict[str, Any], poi_kind: str, destination: str, index: int) -> Any:
        if poi_kind == "attraction":
            return self._attraction_from_candidate(item, destination, index)
        if poi_kind == "dining":
            return self._dining_from_candidate(item)
        if poi_kind == "hotel":
            return self._hotel_from_candidate(item)
        return None

    def _hydrate_place_admin(self, place: Any, admin_index: Dict[str, Dict[str, Any]]) -> None:
        if not place or (getattr(place, "cityname", "") and getattr(place, "adname", "")):
            return
        for key in (
            self._candidate_lookup_key(getattr(place, "name", ""), getattr(place, "location", None)),
            self._candidate_lookup_key(getattr(place, "name", "")),
        ):
            item = admin_index.get(key)
            if not item:
                continue
            for field in ("cityname", "adname", "adcode"):
                if hasattr(place, field) and not getattr(place, field, "") and item.get(field):
                    setattr(place, field, item[field])
            if item.get("location") and getattr(place, "location", None) is None:
                place.location = Location(**item["location"])
            break

    def _validate_and_filter_plan(
        self,
        plan: TripPlanResponse,
        destination: str,
        candidate_context: Optional[Dict[str, Any]] = None,
    ) -> TripPlanResponse:
        admin_index = self._build_candidate_admin_index(candidate_context)
        attraction_candidates = self._candidate_items_by_section(candidate_context, "attractions")
        dining_candidates = self._candidate_items_by_section(candidate_context, "dinings")
        hotel_candidates = self._candidate_items_by_section(candidate_context, "hotels")
        used_attractions = {str(item.name) for day in plan.days for item in day.attractions if item.name}
        used_dinings = {str(item.name) for day in plan.days for item in day.dinings if item.name}
        used_hotels = {str(item.name) for item in plan.hotels if item.name}
        replacement_count = 0
        filtered_days: List[DailyPlan] = []
        for day in plan.days:
            valid_attractions = []
            for index, attraction in enumerate(day.attractions):
                self._hydrate_place_admin(attraction, admin_index)
                if self._needs_candidate_replacement(attraction, attraction_candidates, destination):
                    replacement = self._next_candidate(
                        attraction_candidates,
                        used_attractions,
                        destination=destination,
                        poi_kind="attraction",
                    )
                    if replacement:
                        attraction = self._replace_place_from_candidate(replacement, "attraction", destination, index)
                        replacement_count += 1
                if attraction and self._validate_place_in_destination(attraction, destination):
                    valid_attractions.append(attraction)
            day.attractions = valid_attractions
            valid_dinings = []
            for index, dining in enumerate(day.dinings):
                self._hydrate_place_admin(dining, admin_index)
                if self._needs_candidate_replacement(dining, dining_candidates, destination):
                    replacement = self._next_candidate(
                        dining_candidates,
                        used_dinings,
                        destination=destination,
                        poi_kind="dining",
                    )
                    if replacement:
                        dining = self._replace_place_from_candidate(replacement, "dining", destination, index)
                        replacement_count += 1
                if dining and self._validate_place_in_destination(dining, destination):
                    valid_dinings.append(dining)
            day.dinings = valid_dinings
            if day.recommended_hotel:
                self._hydrate_place_admin(day.recommended_hotel, admin_index)
                if self._needs_candidate_replacement(day.recommended_hotel, hotel_candidates, destination):
                    replacement = self._next_candidate(
                        hotel_candidates,
                        used_hotels,
                        destination=destination,
                        poi_kind="hotel",
                    )
                    if replacement:
                        day.recommended_hotel = self._replace_place_from_candidate(replacement, "hotel", destination, 0)
                        replacement_count += 1
                if day.recommended_hotel and not self._validate_place_in_destination(day.recommended_hotel, destination):
                    day.recommended_hotel = None
            filtered_days.append(day)
        plan.days = filtered_days
        if replacement_count:
            logger.info(
                "Plan POIs repaired with ranked candidates",
                extra={"destination": destination, "replacement_count": replacement_count},
            )
        return plan

    def _build_fallback_plan(self, request: TripPlanRequest) -> TripPlanResponse:
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)
        day_count = max(1, min((end_date - start_date).days + 1, 7))
        hotel = Hotel(
            name=f"{request.destination}市中心经济型酒店",
            address=f"{request.destination}市中心",
            price="经济型",
            rating="N/A",
        )
        days: List[DailyPlan] = []
        for index in range(day_count):
            date_text = (start_date + timedelta(days=index)).date().isoformat()
            days.append(
                DailyPlan(
                    day=index + 1,
                    theme=f"{request.destination}第{index + 1}天轻量行程",
                    weather=Weather(
                        date=date_text,
                        day_weather="请以出行当天实时天气为准",
                        night_weather="请以出行当天实时天气为准",
                        day_temp="N/A",
                        night_temp="N/A",
                    ),
                    recommended_hotel=hotel,
                    attractions=[
                        Attraction(
                            name=f"{request.destination}城市核心游览区",
                            type="城市游览",
                            rating="N/A",
                            suggested_duration_hours=3,
                            description="LangGraph最终规划接口不可用，系统生成临时可测试行程。建议结合地图搜索补充具体景点。",
                            address=f"{request.destination}市区",
                            location=None,
                            ticket_price="N/A",
                        )
                    ],
                    dinings=[
                        Dining(
                            name=f"{request.destination}本地餐饮推荐",
                            address=f"{request.destination}市区",
                            cost_per_person="按实际消费",
                            rating="N/A",
                        )
                    ],
                    budget=DailyBudget(total=0),
                )
            )
        return TripPlanResponse(
            trip_title=f"{request.destination}{day_count}日临时行程",
            total_budget=BudgetBreakdown(total=0),
            hotels=[hotel],
            days=days,
        )
