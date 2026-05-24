from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent

from app.observability.logger import default_logger as logger


class AMapMCPClient:
    """Small stdio MCP client for AMap tools used by the LangGraph planner."""

    def __init__(self) -> None:
        self._exit_stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None

    def _server_params(self) -> StdioServerParameters:
        backend_dir = Path(__file__).resolve().parents[2]
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp.amap_server"],
            cwd=backend_dir,
        )

    async def _ensure_session(self) -> ClientSession:
        if self._session is not None:
            return self._session

        exit_stack = AsyncExitStack()
        read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(self._server_params()))
        session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()

        self._exit_stack = exit_stack
        self._session = session
        logger.info("AMap MCP client initialized")
        return session

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        async with AsyncExitStack() as exit_stack:
            read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(self._server_params()))
            session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            logger.info("Calling AMap MCP tool", extra={"tool": name})
            result = await session.call_tool(name, arguments)
            if result.isError:
                raise RuntimeError(f"MCP tool {name} failed: {result.content}")
            payload = self._decode_tool_result(result)
            logger.info(
                "AMap MCP tool completed",
                extra={"tool": name, "result_count": len(payload) if isinstance(payload, list) else None},
            )
            return payload

    async def search_pois(
        self,
        *,
        keywords: str,
        city: str,
        limit: int,
        category: str,
        citylimit: bool = True,
    ) -> List[Dict[str, Any]]:
        payload = await self.call_tool(
            "search_pois",
            {
                "keywords": keywords,
                "city": city,
                "limit": limit,
                "category": category,
                "citylimit": citylimit,
            },
        )
        return payload if isinstance(payload, list) else []

    async def get_weather_forecast(self, city: str) -> List[Dict[str, Any]]:
        payload = await self.call_tool("get_weather_forecast", {"city": city})
        return payload if isinstance(payload, list) else []

    async def get_route_plan(
        self,
        *,
        origin_lng: float,
        origin_lat: float,
        destination_lng: float,
        destination_lat: float,
        mode: str = "walking",
    ) -> Dict[str, Any]:
        payload = await self.call_tool(
            "get_route_plan",
            {
                "origin_lng": origin_lng,
                "origin_lat": origin_lat,
                "destination_lng": destination_lng,
                "destination_lat": destination_lat,
                "mode": mode,
            },
        )
        return payload if isinstance(payload, dict) else {}

    def _decode_tool_result(self, result: Any) -> Any:
        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return self._decode_tool_content(getattr(result, "content", None))

    def _decode_tool_content(self, content: Any) -> Any:
        if not content:
            return None
        if len(content) > 1:
            decoded_items = []
            for item in content:
                if hasattr(item, "text"):
                    try:
                        decoded_items.append(json.loads(item.text))
                        continue
                    except Exception:
                        decoded_items.append(item.text)
                        continue
                decoded_items.append(item)
            return decoded_items

        first = content[0]
        if isinstance(first, TextContent):
            try:
                return json.loads(first.text)
            except json.JSONDecodeError:
                return first.text
        if hasattr(first, "text"):
            try:
                return json.loads(first.text)
            except Exception:
                return first.text
        return content

    async def close(self) -> None:
        if self._exit_stack is None:
            return
        try:
            await self._exit_stack.aclose()
        finally:
            self._exit_stack = None
            self._session = None


amap_mcp_client = AMapMCPClient()
