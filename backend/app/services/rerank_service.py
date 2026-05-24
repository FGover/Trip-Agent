from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from app.config import settings
from app.observability.logger import default_logger as logger


class RerankService:
    """DashScope/Model Studio rerank client.

    The service keeps retrieval deterministic by returning the original order
    whenever the external rerank API is disabled or unavailable.
    """

    def __init__(self) -> None:
        self.enabled = settings.RERANK_ENABLED
        self.model = settings.RERANK_MODEL
        self.api_key = settings.RERANK_API_KEY or settings.EMBEDDING_API_KEY or settings.LLM_API_KEY
        self.url = settings.RERANK_BASE_URL

    def rerank(
        self,
        *,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not self.api_key or not query.strip() or not documents:
            return self._identity_results(documents)

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n or len(documents), len(documents)),
            "return_documents": False,
        }
        try:
            response = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=min(settings.LLM_TIMEOUT, 30),
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("results") or data.get("output", {}).get("results") or []
            parsed = []
            for result in results:
                index = result.get("index")
                if index is None:
                    continue
                parsed.append(
                    {
                        "index": int(index),
                        "relevance_score": float(result.get("relevance_score") or 0.0),
                    }
                )
            if not parsed:
                logger.warning("Rerank API returned no scored results")
                return self._identity_results(documents)
            logger.info(
                "Cross-encoder rerank completed",
                extra={"model": self.model, "document_count": len(documents), "result_count": len(parsed)},
            )
            return parsed
        except Exception as exc:
            logger.warning(
                "Cross-encoder rerank failed; falling back to rule ranking",
                extra={"error": str(exc), "model": self.model},
            )
            return self._identity_results(documents)

    @staticmethod
    def _identity_results(documents: List[str]) -> List[Dict[str, Any]]:
        return [
            {"index": index, "relevance_score": float(len(documents) - index) / max(len(documents), 1)}
            for index, _ in enumerate(documents)
        ]


rerank_service = RerankService()
