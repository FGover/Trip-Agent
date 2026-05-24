from __future__ import annotations

from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions

from app.config import settings
from app.observability.logger import default_logger as logger


class ElasticsearchMemoryService:
    """BM25 keyword retrieval backed by Elasticsearch inverted indexes."""

    def __init__(self) -> None:
        self.enabled = settings.ELASTICSEARCH_ENABLED
        self.available = False
        self.client: Optional[Elasticsearch] = None
        self.user_index = f"{settings.ELASTICSEARCH_INDEX_PREFIX}_user"
        self.knowledge_index = f"{settings.ELASTICSEARCH_INDEX_PREFIX}_knowledge"
        if not self.enabled:
            logger.info("Elasticsearch memory retrieval disabled")
            return

        try:
            auth = None
            if settings.ELASTICSEARCH_USER and settings.ELASTICSEARCH_PASSWORD:
                auth = (settings.ELASTICSEARCH_USER, settings.ELASTICSEARCH_PASSWORD)
            self.client = Elasticsearch(
                settings.ELASTICSEARCH_URL,
                basic_auth=auth,
                request_timeout=5,
                retry_on_timeout=True,
                max_retries=1,
            )
            self.available = bool(self.client.ping())
            if self.available:
                self._ensure_indexes()
                logger.info("Elasticsearch memory retrieval connected", extra={"url": settings.ELASTICSEARCH_URL})
            else:
                logger.warning("Elasticsearch memory retrieval unavailable", extra={"url": settings.ELASTICSEARCH_URL})
        except Exception as exc:
            self.available = False
            logger.warning("Elasticsearch memory retrieval initialization failed", extra={"error": str(exc)})

    def _ensure_indexes(self) -> None:
        assert self.client is not None
        for index_name in (self.user_index, self.knowledge_index):
            if self.client.indices.exists(index=index_name):
                continue
            self.client.indices.create(
                index=index_name,
                mappings={
                    "properties": {
                        "metadata_id": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                        "type": {"type": "keyword"},
                        "text_representation": {"type": "text", "index": False},
                        "bm25_text": {"type": "text", "analyzer": "standard"},
                        "timestamp": {"type": "date", "ignore_malformed": True},
                        "data": {"type": "object", "enabled": False},
                    }
                },
            )

    def is_available(self) -> bool:
        return self.enabled and self.available and self.client is not None

    def index_memory(
        self,
        *,
        scope: str,
        metadata_id: str,
        metadata: Dict[str, Any],
        bm25_tokens: List[str],
    ) -> None:
        if not self.is_available():
            return
        if metadata.get("is_active", True) is False:
            self.delete_memory(scope=scope, metadata_id=metadata_id)
            return
        index_name = self.user_index if scope == "user" else self.knowledge_index
        document = {
            "metadata_id": metadata_id,
            "user_id": metadata.get("user_id", ""),
            "type": metadata.get("type", ""),
            "text_representation": metadata.get("text_representation", ""),
            "bm25_text": " ".join(bm25_tokens),
            "timestamp": metadata.get("timestamp"),
            "data": metadata.get("data", {}),
        }
        try:
            self.client.index(index=index_name, id=metadata_id, document=document)
        except Exception as exc:
            self.available = False
            logger.warning("Elasticsearch memory index failed", extra={"scope": scope, "error": str(exc)})

    def delete_memory(self, *, scope: str, metadata_id: str) -> None:
        if not self.is_available():
            return
        index_name = self.user_index if scope == "user" else self.knowledge_index
        try:
            self.client.delete(index=index_name, id=metadata_id)
        except es_exceptions.NotFoundError:
            return
        except Exception as exc:
            self.available = False
            logger.warning(
                "Elasticsearch memory delete failed",
                extra={"scope": scope, "metadata_id": metadata_id, "error": str(exc)},
            )

    def bulk_reindex(
        self,
        *,
        scope: str,
        metadata_store: Dict[str, Dict[str, Any]],
        bm25_cache: Dict[str, Dict[str, Any]],
    ) -> None:
        if not self.is_available():
            return
        indexed_count = 0
        for metadata_id, metadata in metadata_store.items():
            if metadata.get("is_active", True) is False:
                self.delete_memory(scope=scope, metadata_id=metadata_id)
                continue
            cached = bm25_cache.get(metadata_id) or {}
            tokens = cached.get("tokens") or []
            self.index_memory(scope=scope, metadata_id=metadata_id, metadata=metadata, bm25_tokens=tokens)
            indexed_count += 1
        try:
            index_name = self.user_index if scope == "user" else self.knowledge_index
            self.client.indices.refresh(index=index_name)
        except Exception:
            pass
        logger.info("Elasticsearch memory index refreshed", extra={"scope": scope, "count": indexed_count})

    def search(
        self,
        *,
        scope: str,
        query_tokens: List[str],
        metadata_store: Dict[str, Dict[str, Any]],
        limit: int,
        user_id: Optional[str] = None,
        allowed_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.is_available() or not query_tokens:
            return []

        filters: List[Dict[str, Any]] = []
        if user_id is not None:
            filters.append({"term": {"user_id": user_id}})
        if allowed_types:
            filters.append({"terms": {"type": allowed_types}})

        index_name = self.user_index if scope == "user" else self.knowledge_index
        try:
            response = self.client.search(
                index=index_name,
                size=limit,
                query={
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "bm25_text": {
                                        "query": " ".join(query_tokens),
                                        "operator": "or",
                                    }
                                }
                            }
                        ],
                        "filter": filters,
                    }
                },
            )
        except es_exceptions.NotFoundError:
            return []
        except Exception as exc:
            self.available = False
            logger.warning("Elasticsearch BM25 search failed", extra={"scope": scope, "error": str(exc)})
            return []

        results: List[Dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            metadata_id = str(hit.get("_id") or hit.get("_source", {}).get("metadata_id") or "")
            metadata = metadata_store.get(metadata_id)
            if not metadata:
                continue
            if metadata.get("is_active", True) is False:
                continue
            result = dict(metadata)
            result["metadata_id"] = metadata_id
            result["bm25_score"] = float(hit.get("_score") or 0.0)
            result["bm25_provider"] = "elasticsearch"
            results.append(result)
        return results


elasticsearch_memory_service = ElasticsearchMemoryService()
