from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests
from mcp.server.fastmcp import FastMCP

from app.config import settings


mcp = FastMCP(
    "wayfinder-amap",
    instructions="AMap tools exposed through MCP for WayfinderAI trip planning.",
)


def _parse_location(location_text: Any) -> Optional[Dict[str, float]]:
    if not isinstance(location_text, str) or "," not in location_text:
        return None
    try:
        lng, lat = location_text.split(",", 1)
        return {"lat": float(lat), "lng": float(lng)}
    except Exception:
        return None


def _normalize_poi(poi: Dict[str, Any], category: str) -> Dict[str, Any]:
    return {
        "name": poi.get("name", ""),
        "type": poi.get("type") or category,
        "address": poi.get("address") if isinstance(poi.get("address"), str) else "",
        "location": _parse_location(poi.get("location")),
        "rating": (poi.get("biz_ext") or {}).get("rating") or poi.get("rating") or "N/A",
        "price": (poi.get("biz_ext") or {}).get("cost") or "N/A",
        "cityname": poi.get("cityname") if isinstance(poi.get("cityname"), str) else "",
        "adname": poi.get("adname") if isinstance(poi.get("adname"), str) else "",
    }


@mcp.tool()
def search_pois(
    keywords: str,
    city: str = "",
    limit: int = 8,
    category: str = "景点",
    citylimit: bool = True,
) -> List[Dict[str, Any]]:
    """Search AMap POIs for attractions, hotels, restaurants, or other places."""
    if not settings.AMAP_API_KEY:
        return []

    bounded_limit = max(1, min(int(limit), 20))
    response = requests.get(
        "https://restapi.amap.com/v3/place/text",
        params={
            "key": settings.AMAP_API_KEY,
            "keywords": keywords,
            "city": city,
            "citylimit": "true" if citylimit else "false",
            "offset": bounded_limit,
            "page": 1,
            "extensions": "all",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    pois = data.get("pois", []) if data.get("status") == "1" else []
    return [_normalize_poi(poi, category) for poi in pois[:bounded_limit]]


@mcp.tool()
def get_weather_forecast(city: str) -> List[Dict[str, Any]]:
    """Get AMap weather forecast for a city."""
    if not settings.AMAP_API_KEY:
        return []

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
        for item in forecasts[0].get("casts", [])
    ]


@mcp.tool()
def get_route_plan(
    origin_lng: float,
    origin_lat: float,
    destination_lng: float,
    destination_lat: float,
    mode: str = "walking",
) -> Dict[str, Any]:
    """Get AMap route distance and duration between two coordinates."""
    if not settings.AMAP_API_KEY:
        return {}

    normalized_mode = mode if mode in {"walking", "driving"} else "walking"
    endpoint = (
        "https://restapi.amap.com/v3/direction/walking"
        if normalized_mode == "walking"
        else "https://restapi.amap.com/v3/direction/driving"
    )
    response = requests.get(
        endpoint,
        params={
            "key": settings.AMAP_API_KEY,
            "origin": f"{origin_lng},{origin_lat}",
            "destination": f"{destination_lng},{destination_lat}",
        },
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    route = data.get("route", {}) if data.get("status") == "1" else {}
    paths = route.get("paths") or []
    if not paths:
        return {}

    first_path = paths[0]
    distance_m = float(first_path.get("distance") or 0)
    duration_s = float(first_path.get("duration") or 0)
    return {
        "mode": normalized_mode,
        "distance_km": round(distance_m / 1000, 2),
        "duration_minutes": round(duration_s / 60, 1),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
