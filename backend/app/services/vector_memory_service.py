"""
基于向量数据库的记忆服务
支持用户记忆、知识记忆的向量存储和语义检索
"""
import json
import math
import os
import re
import threading
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
from openai import OpenAI

try:
    import jieba
except ImportError:  # pragma: no cover - dependency fallback for minimal environments
    jieba = None

from app.observability.logger import default_logger as logger
from app.config import settings
from app.services.elasticsearch_memory_service import elasticsearch_memory_service
from app.services.rerank_service import rerank_service


class VectorMemoryService:
    """
    基于向量数据库的记忆服务（单例模式）
    使用FAISS进行向量存储，DashScope/OpenAI-compatible API进行文本嵌入
    实现线程安全的单例模式，避免重复初始化模型
    """
    
    # 单例实例
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """实现单例模式的 __new__ 方法"""
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定，确保线程安全
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        vector_dim: Optional[int] = None,
        memory_dir: str = "vector_memory"
    ):
        """
        初始化向量记忆服务
        
        Args:
            model_name: 句子嵌入模型名称（如果为None，使用配置中的模型）
            vector_dim: 向量维度
            memory_dir: 记忆存储目录
        
        注意：由于是单例模式，初始化只会执行一次
        """
        # 避免重复初始化
        if hasattr(self, '_initialized') and self._initialized:
            logger.debug("向量记忆服务已初始化，跳过重复初始化")
            return
        
        logger.info("初始化向量记忆服务（单例模式）...")
        
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dim = vector_dim or settings.VECTOR_DIM
        self._save_lock = threading.Lock()
        self._save_timer: Optional[threading.Timer] = None
        self._pending_save = False
        self._save_delay_seconds = 2.0
        self._maintenance_thread: Optional[threading.Thread] = None
        self._maintenance_stop_event = threading.Event()
        self.embedding_provider = settings.EMBEDDING_PROVIDER.lower()
        self.bm25_k1 = 1.5
        self.bm25_b = 0.75
        self.hybrid_vector_weight = 0.7
        self.hybrid_bm25_weight = 0.3
        self.rrf_k = 60
        self.bm25_tokenizer_version = "jieba_search_v2"
        
        # 使用配置中的模型名称（如果没有提供）
        if model_name is None:
            model_name = settings.EMBEDDING_MODEL
            logger.info(f"使用配置中的嵌入模型: {model_name}")
        self.embedding_model_name = model_name
        
        self.embedding_client = self._create_embedding_client()
        self.embedding_model = self.embedding_client
        
        # 初始化FAISS索引
        self.user_memory_index = None
        self.knowledge_memory_index = None
        self.user_metadata = {}  # 存储用户记忆的元数据
        self.knowledge_metadata = {}  # 存储知识记忆的元数据
        self.user_bm25_cache: Dict[str, Dict[str, Any]] = {}
        self.knowledge_bm25_cache: Dict[str, Dict[str, Any]] = {}
        
        # 加载或创建索引
        self._load_or_create_indexes()
        
        # 标记为已初始化
        self._initialized = True
        logger.info("向量记忆服务初始化完成（单例模式）")
    
    def _create_embedding_client(self) -> OpenAI:
        if self.embedding_provider not in {"dashscope", "aliyun", "openai"}:
            raise ValueError(
                f"Unsupported EMBEDDING_PROVIDER '{settings.EMBEDDING_PROVIDER}'. "
                "Use dashscope/aliyun/openai for an OpenAI-compatible embeddings service."
            )

        api_key = settings.EMBEDDING_API_KEY or settings.LLM_API_KEY
        base_url = settings.EMBEDDING_BASE_URL or settings.LLM_BASE_URL
        if not api_key or not base_url:
            raise ValueError(
                "EMBEDDING_API_KEY/LLM_API_KEY and EMBEDDING_BASE_URL/LLM_BASE_URL are required"
            )

        logger.info(
            "Embedding service configured",
            extra={
                "provider": self.embedding_provider,
                "model": self.embedding_model_name,
                "base_url": base_url,
                "vector_dim": self.vector_dim,
            },
        )
        return OpenAI(api_key=api_key, base_url=base_url, timeout=settings.LLM_TIMEOUT)

    def _load_or_create_indexes(self):
        """加载或创建FAISS索引"""
        user_index_path = self.memory_dir / "user_memory.index"
        knowledge_index_path = self.memory_dir / "knowledge_memory.index"
        user_metadata_path = self.memory_dir / "user_metadata.json"
        knowledge_metadata_path = self.memory_dir / "knowledge_metadata.json"
        user_bm25_path = self.memory_dir / "user_bm25_cache.json"
        knowledge_bm25_path = self.memory_dir / "knowledge_bm25_cache.json"
        
        # 尝试加载用户记忆索引
        if user_index_path.exists():
            self.user_memory_index = faiss.read_index(str(user_index_path))
            if self.user_memory_index.d != self.vector_dim:
                logger.warning(
                    "User memory index dimension mismatch; recreating index",
                    extra={"expected": self.vector_dim, "actual": self.user_memory_index.d},
                )
                self.user_memory_index = faiss.IndexFlatIP(self.vector_dim)
                self.user_metadata = {}
            elif user_metadata_path.exists():
                with open(user_metadata_path, 'r', encoding='utf-8') as f:
                    self.user_metadata = json.load(f)
            logger.info(f"已加载用户记忆索引，包含 {self.user_memory_index.ntotal} 条记录")
        else:
            self.user_memory_index = faiss.IndexFlatIP(self.vector_dim)  # 内积相似度
            self.user_metadata = {}
            logger.info("创建了新的用户记忆索引")

        self.user_bm25_cache = self._load_bm25_cache(user_bm25_path)
        self._rebuild_missing_bm25_cache(self.user_metadata, self.user_bm25_cache)
        
        # 尝试加载知识记忆索引
        if knowledge_index_path.exists():
            self.knowledge_memory_index = faiss.read_index(str(knowledge_index_path))
            if self.knowledge_memory_index.d != self.vector_dim:
                logger.warning(
                    "Knowledge memory index dimension mismatch; recreating index",
                    extra={"expected": self.vector_dim, "actual": self.knowledge_memory_index.d},
                )
                self.knowledge_memory_index = faiss.IndexFlatIP(self.vector_dim)
                self.knowledge_metadata = {}
            elif knowledge_metadata_path.exists():
                with open(knowledge_metadata_path, 'r', encoding='utf-8') as f:
                    self.knowledge_metadata = json.load(f)
            logger.info(f"已加载知识记忆索引，包含 {self.knowledge_memory_index.ntotal} 条记录")
        else:
            self.knowledge_memory_index = faiss.IndexFlatIP(self.vector_dim)  # 内积相似度
            self.knowledge_metadata = {}
            logger.info("创建了新的知识记忆索引")
        self.knowledge_bm25_cache = self._load_bm25_cache(knowledge_bm25_path)
        self._rebuild_missing_bm25_cache(self.knowledge_metadata, self.knowledge_bm25_cache)
        self._sync_elasticsearch_indexes()
    
    def _save_indexes(self):
        """保存FAISS索引和元数据"""
        try:
            # 保存用户记忆索引
            user_index_path = self.memory_dir / "user_memory.index"
            user_metadata_path = self.memory_dir / "user_metadata.json"
            user_bm25_path = self.memory_dir / "user_bm25_cache.json"
            faiss.write_index(self.user_memory_index, str(user_index_path))
            with open(user_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_metadata, f, ensure_ascii=False, indent=2)
            with open(user_bm25_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_bm25_cache, f, ensure_ascii=False, indent=2)
            
            # 保存知识记忆索引
            knowledge_index_path = self.memory_dir / "knowledge_memory.index"
            knowledge_metadata_path = self.memory_dir / "knowledge_metadata.json"
            knowledge_bm25_path = self.memory_dir / "knowledge_bm25_cache.json"
            faiss.write_index(self.knowledge_memory_index, str(knowledge_index_path))
            with open(knowledge_metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_metadata, f, ensure_ascii=False, indent=2)
            with open(knowledge_bm25_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_bm25_cache, f, ensure_ascii=False, indent=2)
            
            logger.info("向量索引保存成功")
        except Exception as e:
            logger.error(f"保存向量索引失败: {e}")

    def schedule_save(self, delay_seconds: Optional[float] = None) -> None:
        """Debounce vector index persistence to keep request paths responsive."""
        delay = self._save_delay_seconds if delay_seconds is None else delay_seconds

        with self._save_lock:
            self._pending_save = True
            if self._save_timer:
                self._save_timer.cancel()

            self._save_timer = threading.Timer(delay, self._flush_scheduled_save)
            self._save_timer.daemon = True
            self._save_timer.start()

    def _flush_scheduled_save(self) -> None:
        with self._save_lock:
            self._save_timer = None
            if not self._pending_save:
                return
            self._pending_save = False

        self._save_indexes()
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to a normalized embedding vector via DashScope/OpenAI-compatible API."""
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model_name,
                input=text,
                dimensions=self.vector_dim,
            )
            vector = np.array(response.data[0].embedding, dtype='float32')
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            if len(vector) != self.vector_dim:
                logger.warning(
                    "Embedding dimension mismatch",
                    extra={"expected": self.vector_dim, "actual": len(vector)},
                )
                return np.zeros(self.vector_dim, dtype='float32')
            return vector.astype('float32')
        except Exception as e:
            logger.error(f"文本向量化失败: {e}")
            return np.zeros(self.vector_dim, dtype='float32')

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenize Chinese and mixed text for BM25, using a real segmenter when available."""
        if not text:
            return []

        normalized = text.lower()
        tokens: List[str] = []

        for chunk in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", normalized):
            if re.fullmatch(r"[\u4e00-\u9fff]+", chunk):
                if jieba is not None:
                    tokens.extend(token.strip() for token in jieba.cut_for_search(chunk) if token.strip())
                else:
                    tokens.append(chunk)
            else:
                tokens.append(chunk)

        return list(dict.fromkeys(token for token in tokens if token.strip()))

    def _bm25_text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_bm25_cache(self, path: Path) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning("Failed to load BM25 cache", extra={"path": str(path), "error": str(exc)})
            return {}

    def _build_bm25_cache_entry(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        text = metadata.get("text_representation", "") or ""
        tokens = self._tokenize_for_bm25(text)
        return {
            "text_hash": self._bm25_text_hash(text),
            "tokenizer_version": self.bm25_tokenizer_version,
            "tokens": tokens,
            "doc_len": len(tokens),
        }

    def _rebuild_missing_bm25_cache(
        self,
        metadata_store: Dict[str, Dict[str, Any]],
        cache_store: Dict[str, Dict[str, Any]],
    ) -> None:
        stale_ids = set(cache_store) - set(metadata_store)
        for metadata_id in stale_ids:
            cache_store.pop(metadata_id, None)

        rebuilt_count = 0
        for metadata_id, metadata in metadata_store.items():
            text = metadata.get("text_representation", "") or ""
            text_hash = self._bm25_text_hash(text)
            cached = cache_store.get(metadata_id)
            if (
                cached
                and cached.get("text_hash") == text_hash
                and cached.get("tokenizer_version") == self.bm25_tokenizer_version
                and isinstance(cached.get("tokens"), list)
            ):
                continue
            cache_store[metadata_id] = self._build_bm25_cache_entry(metadata)
            rebuilt_count += 1

        if rebuilt_count or stale_ids:
            logger.info(
                "BM25 cache refreshed",
                extra={"rebuilt": rebuilt_count, "removed": len(stale_ids)},
            )

    def _update_bm25_cache_entry(
        self,
        cache_store: Dict[str, Dict[str, Any]],
        metadata_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        cache_store[metadata_id] = self._build_bm25_cache_entry(metadata)

    def _sync_elasticsearch_indexes(self) -> None:
        if not elasticsearch_memory_service.is_available():
            return
        elasticsearch_memory_service.bulk_reindex(
            scope="user",
            metadata_store=self.user_metadata,
            bm25_cache=self.user_bm25_cache,
        )
        elasticsearch_memory_service.bulk_reindex(
            scope="knowledge",
            metadata_store=self.knowledge_metadata,
            bm25_cache=self.knowledge_bm25_cache,
        )

    def _index_memory_to_elasticsearch(
        self,
        *,
        scope: str,
        metadata_id: str,
        metadata: Dict[str, Any],
        cache_store: Dict[str, Dict[str, Any]],
    ) -> None:
        cached = cache_store.get(metadata_id) or {}
        elasticsearch_memory_service.index_memory(
            scope=scope,
            metadata_id=metadata_id,
            metadata=metadata,
            bm25_tokens=list(cached.get("tokens") or []),
        )

    def _is_memory_active(self, metadata: Dict[str, Any]) -> bool:
        return metadata.get("is_active", True) is not False

    def _delete_memory_from_elasticsearch(self, *, scope: str, metadata_id: str) -> None:
        elasticsearch_memory_service.delete_memory(scope=scope, metadata_id=metadata_id)

    def _mark_memory_inactive(
        self,
        *,
        metadata_id: str,
        metadata: Dict[str, Any],
        reason: str,
    ) -> None:
        metadata["is_active"] = False
        metadata["deleted_at"] = datetime.now().isoformat()
        metadata["delete_reason"] = reason
        self.user_bm25_cache.pop(metadata_id, None)
        self._delete_memory_from_elasticsearch(scope="user", metadata_id=metadata_id)

    def deactivate_trip_memories(self, user_id: str, trip_id: str, reason: str = "trip_changed") -> int:
        deactivated_count = 0
        for metadata_id, metadata in self.user_metadata.items():
            if metadata.get("user_id") != user_id:
                continue
            if not self._is_memory_active(metadata):
                continue
            data = metadata.get("data") if isinstance(metadata.get("data"), dict) else {}
            metadata_trip_id = metadata.get("trip_id") or data.get("trip_id") or data.get("id")
            source_trip_id = metadata.get("source_trip_id") or data.get("source_trip_id")
            if metadata_trip_id != trip_id and source_trip_id != trip_id:
                continue
            self._mark_memory_inactive(
                metadata_id=metadata_id,
                metadata=metadata,
                reason=reason,
            )
            deactivated_count += 1

        if deactivated_count:
            logger.info(
                "Trip memories deactivated",
                extra={
                    "user_id": user_id,
                    "trip_id": trip_id,
                    "reason": reason,
                    "count": deactivated_count,
                },
            )
        return deactivated_count

    def replace_trip_memory(
        self,
        user_id: str,
        trip_id: str,
        trip_data: Dict[str, Any],
        source_event: str = "trip_updated",
    ) -> None:
        self.deactivate_trip_memories(user_id=user_id, trip_id=trip_id, reason=source_event)
        self.store_user_trip(
            user_id=user_id,
            trip_data=trip_data,
            trip_id=trip_id,
            source_event=source_event,
        )

    def deactivate_profile_preferences(self, user_id: str, reason: str = "profile_preferences_replaced") -> int:
        deactivated_count = 0
        for metadata_id, metadata in self.user_metadata.items():
            if metadata.get("user_id") != user_id:
                continue
            if not self._is_memory_active(metadata):
                continue
            if metadata.get("type") != "preference":
                continue
            data = metadata.get("data") if isinstance(metadata.get("data"), dict) else {}
            if metadata.get("preference_type") != "profile_preferences" and data.get("source") != "profile":
                continue
            self._mark_memory_inactive(
                metadata_id=metadata_id,
                metadata=metadata,
                reason=reason,
            )
            deactivated_count += 1
        if deactivated_count:
            logger.info(
                "Profile preference memories deactivated",
                extra={"user_id": user_id, "count": deactivated_count, "reason": reason},
            )
        return deactivated_count

    def replace_profile_preferences(self, user_id: str, preferences: List[str]) -> None:
        self.deactivate_profile_preferences(user_id)
        self.store_user_preference(
            user_id,
            "profile_preferences",
            {
                "preferences": preferences,
                "memory_scope": "long_term",
                "source": "profile",
            },
            source_trip_id=None,
            source_event="profile_updated",
        )

    def rebuild_user_memory_index(self, include_inactive: bool = False) -> Dict[str, int]:
        active_items = [
            (old_id, metadata)
            for old_id, metadata in self.user_metadata.items()
            if include_inactive or self._is_memory_active(metadata)
        ]
        old_total = int(self.user_memory_index.ntotal)
        new_index = faiss.IndexFlatIP(self.vector_dim)
        new_metadata: Dict[str, Dict[str, Any]] = {}
        new_bm25_cache: Dict[str, Dict[str, Any]] = {}

        for new_idx, (_, metadata) in enumerate(active_items):
            text = metadata.get("text_representation", "") or ""
            vector = self._text_to_vector(text)
            new_index.add(np.array([vector]))
            metadata_id = str(new_idx)
            new_metadata[metadata_id] = dict(metadata)
            new_bm25_cache[metadata_id] = self._build_bm25_cache_entry(new_metadata[metadata_id])

        self.user_memory_index = new_index
        self.user_metadata = new_metadata
        self.user_bm25_cache = new_bm25_cache
        self._sync_elasticsearch_indexes()
        self.schedule_save(delay_seconds=0)
        stats = {
            "old_total": old_total,
            "new_total": int(self.user_memory_index.ntotal),
            "removed": old_total - int(self.user_memory_index.ntotal),
        }
        logger.info("User FAISS memory index rebuilt", extra=stats)
        return stats

    def _inactive_user_memory_count(self) -> int:
        return sum(1 for metadata in self.user_metadata.values() if not self._is_memory_active(metadata))

    def rebuild_user_memory_index_if_needed(self, inactive_threshold: Optional[int] = None) -> Optional[Dict[str, int]]:
        threshold = settings.VECTOR_MEMORY_REBUILD_INACTIVE_THRESHOLD if inactive_threshold is None else inactive_threshold
        inactive_count = self._inactive_user_memory_count()
        if inactive_count < threshold:
            logger.debug(
                "User FAISS memory rebuild skipped",
                extra={"inactive_count": inactive_count, "threshold": threshold},
            )
            return None
        logger.info(
            "User FAISS memory rebuild threshold reached",
            extra={"inactive_count": inactive_count, "threshold": threshold},
        )
        return self.rebuild_user_memory_index()

    def start_maintenance_worker(self) -> None:
        interval = settings.VECTOR_MEMORY_REBUILD_INTERVAL_SECONDS
        if interval <= 0:
            logger.info("Vector memory maintenance worker disabled")
            return
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            return

        self._maintenance_stop_event.clear()
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            name="vector-memory-maintenance",
            daemon=True,
        )
        self._maintenance_thread.start()
        logger.info(
            "Vector memory maintenance worker started",
            extra={
                "interval_seconds": interval,
                "inactive_threshold": settings.VECTOR_MEMORY_REBUILD_INACTIVE_THRESHOLD,
            },
        )

    def _maintenance_loop(self) -> None:
        interval = settings.VECTOR_MEMORY_REBUILD_INTERVAL_SECONDS
        while not self._maintenance_stop_event.wait(interval):
            try:
                self.rebuild_user_memory_index_if_needed()
            except Exception as exc:
                logger.warning("Vector memory maintenance failed", extra={"error": str(exc)})

    def stop_maintenance_worker(self) -> None:
        self._maintenance_stop_event.set()

    def _bm25_search(
        self,
        *,
        scope: str,
        query: str,
        metadata_store: Dict[str, Dict[str, Any]],
        cache_store: Dict[str, Dict[str, Any]],
        candidates: Iterable[Tuple[str, Dict[str, Any]]],
        limit: int,
        user_id: Optional[str] = None,
        allowed_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize_for_bm25(query)
        if not query_tokens:
            return []

        es_results = elasticsearch_memory_service.search(
            scope=scope,
            query_tokens=query_tokens,
            metadata_store=metadata_store,
            limit=limit,
            user_id=user_id,
            allowed_types=allowed_types,
        )
        if es_results:
            return es_results

        local_results = self._bm25_rank(query, candidates, limit, cache_store)
        for item in local_results:
            item["bm25_provider"] = "local_cache"
        return local_results

    def _iter_metadata_items(
        self,
        metadata_store: Dict[str, Dict[str, Any]],
        *,
        user_id: Optional[str] = None,
        allowed_types: Optional[List[str]] = None,
    ) -> Iterable[Tuple[str, Dict[str, Any]]]:
        for metadata_id, metadata in metadata_store.items():
            if not self._is_memory_active(metadata):
                continue
            if user_id is not None and metadata.get("user_id") != user_id:
                continue
            if allowed_types and metadata.get("type") not in allowed_types:
                continue
            yield metadata_id, metadata

    def _bm25_rank(
        self,
        query: str,
        candidates: Iterable[Tuple[str, Dict[str, Any]]],
        limit: int,
        cache_store: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize_for_bm25(query)
        if not query_tokens:
            return []

        documents: List[Tuple[str, Dict[str, Any], List[str]]] = []
        document_frequency: Counter[str] = Counter()
        total_length = 0

        for metadata_id, metadata in candidates:
            cached = cache_store.get(metadata_id)
            text = metadata.get("text_representation", "") or ""
            text_hash = self._bm25_text_hash(text)
            if (
                not cached
                or cached.get("text_hash") != text_hash
                or cached.get("tokenizer_version") != self.bm25_tokenizer_version
                or not isinstance(cached.get("tokens"), list)
            ):
                cached = self._build_bm25_cache_entry(metadata)
                cache_store[metadata_id] = cached
            tokens = list(cached.get("tokens") or [])
            if not tokens:
                continue
            documents.append((metadata_id, metadata, tokens))
            total_length += len(tokens)
            document_frequency.update(set(tokens))

        if not documents:
            return []

        avg_doc_len = total_length / len(documents)
        scored: List[Dict[str, Any]] = []
        for metadata_id, metadata, tokens in documents:
            term_frequency = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0

            for token in query_tokens:
                tf = term_frequency.get(token, 0)
                if tf == 0:
                    continue
                df = document_frequency.get(token, 0)
                idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
                denominator = tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_len / avg_doc_len)
                score += idf * (tf * (self.bm25_k1 + 1)) / denominator

            if score <= 0:
                continue

            result = dict(metadata)
            result["metadata_id"] = metadata_id
            result["bm25_score"] = float(score)
            scored.append(result)

        scored.sort(key=lambda item: item.get("bm25_score", 0.0), reverse=True)
        return scored[:limit]

    def _normalize_scores(self, results: List[Dict[str, Any]], score_key: str, normalized_key: str) -> None:
        if not results:
            return
        scores = [float(item.get(score_key, 0.0)) for item in results]
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            for item in results:
                item[normalized_key] = 1.0 if max_score > 0 else 0.0
            return
        for item in results:
            item[normalized_key] = (float(item.get(score_key, 0.0)) - min_score) / (max_score - min_score)

    def _dynamic_hybrid_weights(self, query: str) -> Tuple[float, float]:
        tokens = self._tokenize_for_bm25(query)
        has_specific_terms = any(len(token) >= 2 for token in tokens)
        has_constraints = any(
            marker in query
            for marker in ["不想", "不要", "避免", "必须", "尽量", "靠近", "一定", "优先", "想去"]
        )
        has_long_requirement = len(query) >= 30 or len(tokens) >= 10

        if has_constraints and has_long_requirement:
            return 0.55, 0.45
        if has_specific_terms:
            return 0.65, 0.35
        return 0.8, 0.2

    def _attach_rank_features(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
    ) -> None:
        for rank, item in enumerate(vector_results, start=1):
            item["vector_rank"] = rank
            item["rrf_vector_score"] = 1.0 / (self.rrf_k + rank)
        for rank, item in enumerate(bm25_results, start=1):
            item["bm25_rank"] = rank
            item["rrf_bm25_score"] = 1.0 / (self.rrf_k + rank)

    def _lexical_overlap_score(self, query: str, text: str) -> float:
        query_tokens = set(self._tokenize_for_bm25(query))
        text_tokens = set(self._tokenize_for_bm25(text))
        if not query_tokens or not text_tokens:
            return 0.0
        return len(query_tokens & text_tokens) / len(query_tokens)

    def _freshness_score(self, timestamp: str) -> float:
        if not timestamp:
            return 0.0
        try:
            created_at = datetime.fromisoformat(timestamp)
        except ValueError:
            return 0.0

        age_days = max(0, (datetime.now() - created_at).days)
        return 1.0 / (1.0 + age_days / 30.0)

    def _business_rerank_score(self, item: Dict[str, Any], query: str) -> float:
        text = item.get("text_representation", "")
        overlap = self._lexical_overlap_score(query, text)
        freshness = self._freshness_score(item.get("timestamp", ""))
        type_bonus = 0.05 if item.get("type") == "preference" else 0.03 if item.get("type") == "trip" else 0.0
        return 0.12 * overlap + 0.03 * freshness + type_bonus

    def _fuse_hybrid_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        limit: int,
        query: str,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        self._normalize_scores(vector_results, "similarity_score", "vector_score_norm")
        self._normalize_scores(bm25_results, "bm25_score", "bm25_score_norm")
        self._attach_rank_features(vector_results, bm25_results)

        if vector_weight is None or bm25_weight is None:
            vector_weight, bm25_weight = self._dynamic_hybrid_weights(query)

        merged: Dict[str, Dict[str, Any]] = {}
        for item in vector_results:
            key = item.get("metadata_id") or self._memory_identity(item)
            merged[key] = dict(item)
            merged[key]["retrieval_source"] = "vector"

        for item in bm25_results:
            key = item.get("metadata_id") or self._memory_identity(item)
            if key not in merged:
                merged[key] = dict(item)
                merged[key]["retrieval_source"] = "bm25"
            else:
                merged[key].update(
                    {
                        "bm25_score": item.get("bm25_score", 0.0),
                        "bm25_score_norm": item.get("bm25_score_norm", 0.0),
                        "bm25_rank": item.get("bm25_rank"),
                        "rrf_bm25_score": item.get("rrf_bm25_score", 0.0),
                        "retrieval_source": "hybrid",
                    }
                )

        fused = []
        for item in merged.values():
            vector_score = float(item.get("vector_score_norm", 0.0))
            bm25_score = float(item.get("bm25_score_norm", 0.0))
            rrf_score = float(item.get("rrf_vector_score", 0.0)) + float(item.get("rrf_bm25_score", 0.0))
            business_score = self._business_rerank_score(item, query)
            item["hybrid_score"] = (
                vector_weight * vector_score
                + bm25_weight * bm25_score
            )
            item["rrf_score"] = rrf_score
            item["rerank_score"] = item["hybrid_score"] + rrf_score + business_score
            item["business_rerank_score"] = business_score
            item["hybrid_weights"] = {"vector": vector_weight, "bm25": bm25_weight}
            fused.append(item)

        fused.sort(
            key=lambda item: (
                item.get("rerank_score", 0.0),
                item.get("rrf_score", 0.0),
                item.get("hybrid_score", 0.0),
                item.get("similarity_score", 0.0),
                item.get("bm25_score", 0.0),
            ),
            reverse=True,
        )
        return fused[:limit]

    def _rrf_fuse_recalled_memories(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        candidate_limit: int,
        multi_query_vector_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        rrf_entries: List[Dict[str, Any]] = []
        for rank, item in enumerate(vector_results, start=1):
            entry = dict(item)
            entry["fusion_key"] = item.get("metadata_id") or self._memory_identity(item)
            entry["recall_channel"] = "vector"
            entry["vector_rank"] = rank
            entry["rrf_vector_score"] = 1.0 / (self.rrf_k + rank)
            entry["rrf_bm25_score"] = 0.0
            entry["rrf_score"] = entry["rrf_vector_score"]
            rrf_entries.append(entry)

        for rank, item in enumerate(bm25_results, start=1):
            entry = dict(item)
            entry["fusion_key"] = item.get("metadata_id") or self._memory_identity(item)
            entry["recall_channel"] = "bm25"
            entry["bm25_rank"] = rank
            entry["rrf_vector_score"] = 0.0
            entry["rrf_bm25_score"] = 1.0 / (self.rrf_k + rank)
            entry["rrf_score"] = entry["rrf_bm25_score"]
            rrf_entries.append(entry)

        for rank, item in enumerate(multi_query_vector_results or [], start=1):
            entry = dict(item)
            entry["fusion_key"] = item.get("metadata_id") or self._memory_identity(item)
            entry["recall_channel"] = "multi_query_vector"
            entry["multi_query_vector_rank"] = rank
            entry["rrf_vector_score"] = 0.0
            entry["rrf_bm25_score"] = 0.0
            entry["rrf_multi_query_score"] = 1.0 / (self.rrf_k + rank)
            entry["rrf_score"] = entry["rrf_multi_query_score"]
            rrf_entries.append(entry)

        merged: Dict[str, Dict[str, Any]] = {}
        for entry in rrf_entries:
            key = entry["fusion_key"]
            if key not in merged:
                merged[key] = dict(entry)
                merged[key]["retrieval_source"] = entry["recall_channel"]
                continue

            target = merged[key]
            target["retrieval_source"] = "hybrid"
            target["rrf_vector_score"] = float(target.get("rrf_vector_score", 0.0)) + float(
                entry.get("rrf_vector_score", 0.0)
            )
            target["rrf_bm25_score"] = float(target.get("rrf_bm25_score", 0.0)) + float(
                entry.get("rrf_bm25_score", 0.0)
            )
            target["rrf_multi_query_score"] = float(target.get("rrf_multi_query_score", 0.0)) + float(
                entry.get("rrf_multi_query_score", 0.0)
            )
            target["rrf_score"] = float(target.get("rrf_score", 0.0)) + float(entry.get("rrf_score", 0.0))
            if entry.get("vector_rank") is not None:
                target["vector_rank"] = entry["vector_rank"]
                target["similarity_score"] = entry.get("similarity_score", target.get("similarity_score", 0.0))
            if entry.get("bm25_rank") is not None:
                target["bm25_rank"] = entry["bm25_rank"]
                target["bm25_score"] = entry.get("bm25_score", target.get("bm25_score", 0.0))
                target["bm25_provider"] = entry.get("bm25_provider", target.get("bm25_provider", ""))
            if entry.get("multi_query_vector_rank") is not None:
                target["multi_query_vector_rank"] = entry["multi_query_vector_rank"]
                target["multi_query_source_queries"] = list(
                    set((target.get("multi_query_source_queries") or []) + (entry.get("multi_query_source_queries") or []))
                )
            if not target.get("details") and entry.get("details"):
                target["details"] = entry["details"]

        fused = []
        for item in merged.values():
            item.pop("fusion_key", None)
            item.pop("recall_channel", None)
            item["best_recall_rank"] = min(
                item.get("vector_rank") or 10**9,
                item.get("bm25_rank") or 10**9,
                item.get("multi_query_vector_rank") or 10**9,
            )
            fused.append(item)

        fused.sort(
            key=lambda item: (
                item.get("rrf_score", 0.0),
                -item.get("best_recall_rank", 10**9),
                item.get("similarity_score", 0.0),
                item.get("bm25_score", 0.0),
            ),
            reverse=True,
        )
        return fused[:candidate_limit]

    def _memory_to_rerank_document(self, item: Dict[str, Any]) -> str:
        content = item.get("text_representation") or ""
        details = item.get("details")
        if isinstance(details, dict):
            detail_text = json.dumps(details, ensure_ascii=False)
        elif details:
            detail_text = str(details)
        else:
            detail_text = ""
        parts = [
            f"类型: {item.get('type') or 'memory'}",
            f"内容: {content}",
        ]
        if detail_text:
            parts.append(f"结构化信息: {detail_text[:800]}")
        return "\n".join(parts)

    def _rerank_recalled_memories(
        self,
        *,
        query: str,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        limit: int,
        memory_scope: str,
        multi_query_vector_results: Optional[List[Dict[str, Any]]] = None,
        multi_query_count: int = 0,
    ) -> List[Dict[str, Any]]:
        candidate_limit = max(limit * 4, limit)
        candidates = self._rrf_fuse_recalled_memories(
            vector_results,
            bm25_results,
            candidate_limit,
            multi_query_vector_results=multi_query_vector_results,
        )
        if not candidates:
            return []

        documents = [self._memory_to_rerank_document(item) for item in candidates]
        rerank_results = rerank_service.rerank(
            query=query,
            documents=documents,
            top_n=min(len(documents), max(limit * 4, limit)),
        )
        score_by_index = {
            result["index"]: result["relevance_score"]
            for result in rerank_results
            if 0 <= result.get("index", -1) < len(candidates)
        }
        for index, item in enumerate(candidates):
            item["semantic_rerank_score"] = score_by_index.get(index, 0.0)
            item["final_rerank_score"] = item["semantic_rerank_score"]
            item["ranking_strategy"] = (
                "three_way_faiss_bm25_multi_query_faiss_rrf_then_semantic_rerank"
                if multi_query_vector_results
                else "faiss_bm25_rrf_fusion_then_semantic_rerank"
            )

        candidates.sort(key=lambda item: item.get("final_rerank_score", 0.0), reverse=True)
        logger.info(
            "Memory recall RRF-fused and reranked",
            extra={
                "scope": memory_scope,
                "query": query,
                "vector_count": len(vector_results),
                "bm25_count": len(bm25_results),
                "multi_query_vector_count": len(multi_query_vector_results or []),
                "multi_query_count": multi_query_count,
                "candidate_count": len(candidates),
                "result_count": min(limit, len(candidates)),
                "rrf_k": self.rrf_k,
                "rerank_model": settings.RERANK_MODEL if settings.RERANK_ENABLED else "disabled",
                "top_items": [
                    {
                        "type": item.get("type"),
                        "score": item.get("semantic_rerank_score"),
                        "rrf_score": item.get("rrf_score"),
                        "source": item.get("retrieval_source"),
                        "text": str(item.get("text_representation") or "")[:80],
                    }
                    for item in candidates[:5]
                ],
            },
        )
        return candidates[:limit]

    def _memory_identity(self, metadata: Dict[str, Any]) -> str:
        return "|".join(
            [
                str(metadata.get("user_id", "")),
                str(metadata.get("type", "")),
                str(metadata.get("timestamp", "")),
                str(metadata.get("text_representation", "")),
            ]
        )

    def _vector_to_text(self, vector: np.ndarray) -> str:
        """将向量转换为文本表示（用于调试）"""
        return f"Vector(dim={len(vector)}, norm={np.linalg.norm(vector):.4f})"
    
    # ============ 用户记忆操作 ============
    
    def store_user_preference(
        self,
        user_id: str,
        preference_type: str,
        preference_data: Dict[str, Any],
        source_trip_id: Optional[str] = None,
        source_event: str = "trip_created",
    ):
        """
        存储用户偏好到向量数据库
        
        Args:
            user_id: 用户ID
            preference_type: 偏好类型
            preference_data: 偏好数据
        """
        try:
            # 构建文本表示
            text_representation = self._preference_to_text(preference_type, preference_data)
            
            # 转换为向量
            vector = self._text_to_vector(text_representation)
            
            # 添加到索引
            index_id = self.user_memory_index.ntotal
            self.user_memory_index.add(np.array([vector]))
            
            # 存储元数据
            metadata_id = str(index_id)
            self.user_metadata[metadata_id] = {
                "user_id": user_id,
                "type": "preference",
                "preference_type": preference_type,
                "source_trip_id": source_trip_id,
                "source_event": source_event,
                "is_active": True,
                "data": preference_data,
                "text_representation": text_representation,
                "timestamp": datetime.now().isoformat()
            }
            self._update_bm25_cache_entry(self.user_bm25_cache, metadata_id, self.user_metadata[metadata_id])
            self._index_memory_to_elasticsearch(
                scope="user",
                metadata_id=metadata_id,
                metadata=self.user_metadata[metadata_id],
                cache_store=self.user_bm25_cache,
            )
            
            logger.info(f"用户偏好已存储到向量数据库 - UserID: {user_id}, Type: {preference_type}")
        except Exception as e:
            logger.error(f"存储用户偏好失败: {e}")
    
    def store_user_trip(
        self,
        user_id: str,
        trip_data: Dict[str, Any],
        trip_id: Optional[str] = None,
        source_event: str = "trip_created",
    ):
        """
        存储用户行程到向量数据库
        
        Args:
            user_id: 用户ID
            trip_data: 行程数据
        """
        try:
            # 构建文本表示
            if trip_id:
                trip_data = dict(trip_data)
                trip_data["trip_id"] = trip_id
            text_representation = self._trip_to_text(trip_data)
            
            # 转换为向量
            vector = self._text_to_vector(text_representation)
            
            # 添加到索引
            index_id = self.user_memory_index.ntotal
            self.user_memory_index.add(np.array([vector]))
            
            # 存储元数据
            metadata_id = str(index_id)
            self.user_metadata[metadata_id] = {
                "user_id": user_id,
                "type": "trip",
                "trip_id": trip_id or trip_data.get("trip_id") or trip_data.get("id"),
                "source_event": source_event,
                "version": trip_data.get("version"),
                "is_active": True,
                "data": trip_data,
                "text_representation": text_representation,
                "timestamp": datetime.now().isoformat()
            }
            self._update_bm25_cache_entry(self.user_bm25_cache, metadata_id, self.user_metadata[metadata_id])
            self._index_memory_to_elasticsearch(
                scope="user",
                metadata_id=metadata_id,
                metadata=self.user_metadata[metadata_id],
                cache_store=self.user_bm25_cache,
            )
            
            logger.info(f"用户行程已存储到向量数据库 - UserID: {user_id}")
        except Exception as e:
            logger.error(f"存储用户行程失败: {e}")
    
    def store_user_feedback(
        self,
        user_id: str,
        trip_id: str,
        feedback_data: Dict[str, Any]
    ):
        """
        存储用户反馈到向量数据库
        
        Args:
            user_id: 用户ID
            trip_id: 行程ID
            feedback_data: 反馈数据
        """
        try:
            # 构建文本表示
            text_representation = self._feedback_to_text(feedback_data)
            
            # 转换为向量
            vector = self._text_to_vector(text_representation)
            
            # 添加到索引
            index_id = self.user_memory_index.ntotal
            self.user_memory_index.add(np.array([vector]))
            
            # 存储元数据
            metadata_id = str(index_id)
            self.user_metadata[metadata_id] = {
                "user_id": user_id,
                "type": "feedback",
                "trip_id": trip_id,
                "data": feedback_data,
                "text_representation": text_representation,
                "timestamp": datetime.now().isoformat()
            }
            self._update_bm25_cache_entry(self.user_bm25_cache, metadata_id, self.user_metadata[metadata_id])
            self._index_memory_to_elasticsearch(
                scope="user",
                metadata_id=metadata_id,
                metadata=self.user_metadata[metadata_id],
                cache_store=self.user_bm25_cache,
            )
            
            logger.info(f"用户反馈已存储到向量数据库 - UserID: {user_id}, TripID: {trip_id}")
        except Exception as e:
            logger.error(f"存储用户反馈失败: {e}")
    
    def retrieve_user_memories(
        self,
        user_id: str,
        query: str = "",
        limit: int = 10,
        memory_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        基于语义相似度检索用户记忆
        
        Args:
            user_id: 用户ID
            query: 查询文本
            limit: 返回数量限制
            memory_types: 记忆类型过滤，如 ["preference", "trip", "feedback"]
        
        Returns:
            相似的用户记忆列表
        """
        try:
            # 如果没有查询文本，返回用户最近的记忆
            if not query:
                return self._get_recent_user_memories(user_id, limit, memory_types)
            
            # 转换查询为向量
            query_vector = self._text_to_vector(query)
            
            # 在用户记忆中搜索
            distances, indices = self.user_memory_index.search(
                np.array([query_vector]), 
                min(self.user_memory_index.ntotal, limit * 2)  # 搜索更多结果进行过滤
            )
            
            # 过滤结果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS返回-1表示无效结果
                    continue
                    
                metadata = self.user_metadata.get(str(idx))
                if not metadata:
                    continue

                if not self._is_memory_active(metadata):
                    continue
                
                # 过滤用户ID和记忆类型
                if metadata.get("user_id") != user_id:
                    continue
                    
                if memory_types and metadata.get("type") not in memory_types:
                    continue
                
                result = dict(metadata)
                result["metadata_id"] = str(idx)
                result["similarity_score"] = float(distances[0][i])
                results.append(result)
                
                if len(results) >= limit:
                    break
            
            logger.info(f"检索到 {len(results)} 条用户记忆 - UserID: {user_id}, Query: {query}")
            return results
        except Exception as e:
            logger.error(f"检索用户记忆失败: {e}")
            return []

    def retrieve_user_memories_hybrid(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        memory_types: Optional[List[str]] = None,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not query:
            return self._get_recent_user_memories(user_id, limit, memory_types)

        recall_limit = max(limit * 4, 20)
        vector_results = self.retrieve_user_memories(
            user_id=user_id,
            query=query,
            limit=recall_limit,
            memory_types=memory_types,
        )
        bm25_results = self._bm25_search(
            scope="user",
            query=query,
            metadata_store=self.user_metadata,
            cache_store=self.user_bm25_cache,
            candidates=self._iter_metadata_items(
                self.user_metadata,
                user_id=user_id,
                allowed_types=memory_types,
            ),
            limit=recall_limit,
            user_id=user_id,
            allowed_types=memory_types,
        )
        return self._rerank_recalled_memories(
            query=query,
            vector_results=vector_results,
            bm25_results=bm25_results,
            limit=limit,
            memory_scope="user",
        )

    def retrieve_user_memories_three_way(
        self,
        user_id: str,
        query: str,
        expanded_queries: Optional[List[str]] = None,
        limit: int = 10,
        memory_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not query:
            return self._get_recent_user_memories(user_id, limit, memory_types)

        recall_limit = max(limit * 4, 20)
        vector_results = self.retrieve_user_memories(
            user_id=user_id,
            query=query,
            limit=recall_limit,
            memory_types=memory_types,
        )
        bm25_results = self._bm25_search(
            scope="user",
            query=query,
            metadata_store=self.user_metadata,
            cache_store=self.user_bm25_cache,
            candidates=self._iter_metadata_items(
                self.user_metadata,
                user_id=user_id,
                allowed_types=memory_types,
            ),
            limit=recall_limit,
            user_id=user_id,
            allowed_types=memory_types,
        )
        multi_query_vector_results = self._retrieve_user_memories_by_expanded_faiss(
            user_id=user_id,
            expanded_queries=expanded_queries or [],
            limit=recall_limit,
            memory_types=memory_types,
        )
        return self._rerank_recalled_memories(
            query=query,
            vector_results=vector_results,
            bm25_results=bm25_results,
            multi_query_vector_results=multi_query_vector_results,
            multi_query_count=len(expanded_queries or []),
            limit=limit,
            memory_scope="user",
        )

    def _retrieve_user_memories_by_expanded_faiss(
        self,
        *,
        user_id: str,
        expanded_queries: List[str],
        limit: int,
        memory_types: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        results: List[Dict[str, Any]] = []
        for query in expanded_queries:
            for item in self.retrieve_user_memories(
                user_id=user_id,
                query=query,
                limit=limit,
                memory_types=memory_types,
            ):
                key = item.get("metadata_id") or self._memory_identity(item)
                if key in seen:
                    continue
                seen.add(key)
                enriched = dict(item)
                enriched["multi_query_source_queries"] = [query]
                results.append(enriched)
        return results
    
    def _get_recent_user_memories(
        self,
        user_id: str,
        limit: int,
        memory_types: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """获取用户最近的记忆"""
        user_memories = []
        for metadata in self.user_metadata.values():
            if metadata.get("user_id") != user_id:
                continue

            if not self._is_memory_active(metadata):
                continue
                
            if memory_types and metadata.get("type") not in memory_types:
                continue
                
            user_memories.append(metadata)
        
        # 按时间戳排序
        user_memories.sort(
            key=lambda x: x.get("timestamp", ""), 
            reverse=True
        )
        
        return user_memories[:limit]
    
    # ============ 知识记忆操作 ============
    
    def store_destination_knowledge(
        self,
        destination: str,
        knowledge_data: Dict[str, Any]
    ):
        """
        存储目的地知识到向量数据库
        
        Args:
            destination: 目的地名称
            knowledge_data: 知识数据
        """
        try:
            # 构建文本表示
            text_representation = self._destination_knowledge_to_text(destination, knowledge_data)
            
            # 转换为向量
            vector = self._text_to_vector(text_representation)
            
            # 添加到索引
            index_id = self.knowledge_memory_index.ntotal
            self.knowledge_memory_index.add(np.array([vector]))
            
            # 存储元数据
            metadata_id = str(index_id)
            self.knowledge_metadata[metadata_id] = {
                "type": "destination",
                "destination": destination,
                "data": knowledge_data,
                "text_representation": text_representation,
                "timestamp": datetime.now().isoformat()
            }
            self._update_bm25_cache_entry(self.knowledge_bm25_cache, metadata_id, self.knowledge_metadata[metadata_id])
            self._index_memory_to_elasticsearch(
                scope="knowledge",
                metadata_id=metadata_id,
                metadata=self.knowledge_metadata[metadata_id],
                cache_store=self.knowledge_bm25_cache,
            )
            
            logger.info(f"目的地知识已存储到向量数据库 - Destination: {destination}")
        except Exception as e:
            logger.error(f"存储目的地知识失败: {e}")
    
    def store_travel_experience(
        self,
        experience_type: str,
        experience_data: Dict[str, Any]
    ):
        """
        存储旅行经验到向量数据库
        
        Args:
            experience_type: 经验类型
            experience_data: 经验数据
        """
        try:
            # 构建文本表示
            text_representation = self._experience_to_text(experience_type, experience_data)
            
            # 转换为向量
            vector = self._text_to_vector(text_representation)
            
            # 添加到索引
            index_id = self.knowledge_memory_index.ntotal
            self.knowledge_memory_index.add(np.array([vector]))
            
            # 存储元数据
            metadata_id = str(index_id)
            self.knowledge_metadata[metadata_id] = {
                "type": "experience",
                "experience_type": experience_type,
                "data": experience_data,
                "text_representation": text_representation,
                "timestamp": datetime.now().isoformat()
            }
            self._update_bm25_cache_entry(self.knowledge_bm25_cache, metadata_id, self.knowledge_metadata[metadata_id])
            self._index_memory_to_elasticsearch(
                scope="knowledge",
                metadata_id=metadata_id,
                metadata=self.knowledge_metadata[metadata_id],
                cache_store=self.knowledge_bm25_cache,
            )
            
            logger.info(f"旅行经验已存储到向量数据库 - Type: {experience_type}")
        except Exception as e:
            logger.error(f"存储旅行经验失败: {e}")
    
    def retrieve_knowledge_memories(
        self,
        query: str,
        limit: int = 10,
        knowledge_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        基于语义相似度检索知识记忆
        
        Args:
            query: 查询文本
            limit: 返回数量限制
            knowledge_types: 知识类型过滤，如 ["destination", "experience"]
        
        Returns:
            相似的知识记忆列表
        """
        try:
            # 检查索引是否为空
            if self.knowledge_memory_index.ntotal == 0:
                logger.info(f"知识记忆索引为空，返回空结果 - Query: {query}")
                return []
            
            # 转换查询为向量
            query_vector = self._text_to_vector(query)
            
            # 在知识记忆中搜索
            # 确保k值至少为1，避免FAISS在k=0时报错
            k = min(self.knowledge_memory_index.ntotal, limit * 2)
            distances, indices = self.knowledge_memory_index.search(
                np.array([query_vector]),
                k
            )
            
            # 过滤结果
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1:  # FAISS返回-1表示无效结果
                    continue
                    
                metadata = self.knowledge_metadata.get(str(idx))
                if not metadata:
                    continue
                
                # 过滤知识类型
                if knowledge_types and metadata.get("type") not in knowledge_types:
                    continue
                
                result = dict(metadata)
                result["metadata_id"] = str(idx)
                result["similarity_score"] = float(distances[0][i])
                results.append(result)
                
                if len(results) >= limit:
                    break
            
            logger.info(f"检索到 {len(results)} 条知识记忆 - Query: {query}")
            return results
        except Exception as e:
            logger.error(f"检索知识记忆失败: {e}")
            return []

    def retrieve_knowledge_memories_hybrid(
        self,
        query: str,
        limit: int = 10,
        knowledge_types: Optional[List[str]] = None,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        recall_limit = max(limit * 4, 20)
        vector_results = self.retrieve_knowledge_memories(
            query=query,
            limit=recall_limit,
            knowledge_types=knowledge_types,
        )
        bm25_results = self._bm25_search(
            scope="knowledge",
            query=query,
            metadata_store=self.knowledge_metadata,
            cache_store=self.knowledge_bm25_cache,
            candidates=self._iter_metadata_items(
                self.knowledge_metadata,
                allowed_types=knowledge_types,
            ),
            limit=recall_limit,
            allowed_types=knowledge_types,
        )
        return self._rerank_recalled_memories(
            query=query,
            vector_results=vector_results,
            bm25_results=bm25_results,
            limit=limit,
            memory_scope="knowledge",
        )

    def retrieve_knowledge_memories_three_way(
        self,
        query: str,
        expanded_queries: Optional[List[str]] = None,
        limit: int = 10,
        knowledge_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        recall_limit = max(limit * 4, 20)
        vector_results = self.retrieve_knowledge_memories(
            query=query,
            limit=recall_limit,
            knowledge_types=knowledge_types,
        )
        bm25_results = self._bm25_search(
            scope="knowledge",
            query=query,
            metadata_store=self.knowledge_metadata,
            cache_store=self.knowledge_bm25_cache,
            candidates=self._iter_metadata_items(
                self.knowledge_metadata,
                allowed_types=knowledge_types,
            ),
            limit=recall_limit,
            allowed_types=knowledge_types,
        )
        multi_query_vector_results = self._retrieve_knowledge_memories_by_expanded_faiss(
            expanded_queries=expanded_queries or [],
            limit=recall_limit,
            knowledge_types=knowledge_types,
        )
        return self._rerank_recalled_memories(
            query=query,
            vector_results=vector_results,
            bm25_results=bm25_results,
            multi_query_vector_results=multi_query_vector_results,
            multi_query_count=len(expanded_queries or []),
            limit=limit,
            memory_scope="knowledge",
        )

    def _retrieve_knowledge_memories_by_expanded_faiss(
        self,
        *,
        expanded_queries: List[str],
        limit: int,
        knowledge_types: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        results: List[Dict[str, Any]] = []
        for query in expanded_queries:
            for item in self.retrieve_knowledge_memories(
                query=query,
                limit=limit,
                knowledge_types=knowledge_types,
            ):
                key = item.get("metadata_id") or self._memory_identity(item)
                if key in seen:
                    continue
                seen.add(key)
                enriched = dict(item)
                enriched["multi_query_source_queries"] = [query]
                results.append(enriched)
        return results
    
    # ============ 混合检索 ============
    
    def hybrid_search(
        self,
        user_id: str,
        query: str,
        user_limit: int = 5,
        knowledge_limit: int = 5,
        include_user_memories: bool = True,
        include_knowledge_memories: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Hybrid memory retrieval with FAISS semantic search and BM25 lexical search.
        
        Args:
            user_id: 用户ID
            query: 查询文本
            user_limit: 用户记忆数量限制
            knowledge_limit: 知识记忆数量限制
            include_user_memories: 是否包含用户记忆
            include_knowledge_memories: 是否包含知识记忆
        
        Returns:
            包含用户记忆和知识记忆的字典
        """
        results = {
            "user_memories": [],
            "knowledge_memories": []
        }
        
        # 检索用户记忆
        if include_user_memories:
            results["user_memories"] = self.retrieve_user_memories_hybrid(
                user_id, query, user_limit
            )
        
        # 检索知识记忆
        if include_knowledge_memories:
            results["knowledge_memories"] = self.retrieve_knowledge_memories_hybrid(
                query, knowledge_limit
            )
        
        return results
    
    # ============ 文本转换辅助方法 ============
    
    def _preference_to_text(self, preference_type: str, preference_data: Dict[str, Any]) -> str:
        """将偏好数据转换为文本表示"""
        text_parts = [f"偏好类型: {preference_type}"]
        
        if "destination" in preference_data:
            text_parts.append(f"目的地: {preference_data['destination']}")
        
        if "preferences" in preference_data:
            text_parts.append(f"旅行偏好: {', '.join(preference_data['preferences'])}")
        
        if "hotel_preferences" in preference_data:
            text_parts.append(f"酒店偏好: {', '.join(preference_data['hotel_preferences'])}")
        
        if "budget" in preference_data:
            text_parts.append(f"预算水平: {preference_data['budget']}")

        if "special_requirements" in preference_data and preference_data["special_requirements"]:
            text_parts.append(f"补充需求: {preference_data['special_requirements']}")
        
        return " ".join(text_parts)
    
    def _trip_to_text(self, trip_data: Dict[str, Any]) -> str:
        """将行程数据转换为文本表示"""
        text_parts = ["旅行行程"]
        
        if "destination" in trip_data:
            text_parts.append(f"目的地: {trip_data['destination']}")
        
        if "start_date" in trip_data and "end_date" in trip_data:
            text_parts.append(f"时间: {trip_data['start_date']} 到 {trip_data['end_date']}")
        
        if "preferences" in trip_data:
            text_parts.append(f"偏好: {', '.join(trip_data['preferences'])}")
        
        if "trip_title" in trip_data:
            text_parts.append(f"行程标题: {trip_data['trip_title']}")

        if "special_requirements" in trip_data and trip_data["special_requirements"]:
            text_parts.append(f"补充需求: {trip_data['special_requirements']}")
        
        # 添加景点信息
        if "days" in trip_data:
            attractions = []
            for day in trip_data["days"]:
                for attraction in day.get("attractions", []):
                    attractions.append(attraction.get("name", ""))
            if attractions:
                text_parts.append(f"景点: {', '.join(attractions)}")
        
        return " ".join(text_parts)
    
    def _feedback_to_text(self, feedback_data: Dict[str, Any]) -> str:
        """将反馈数据转换为文本表示"""
        text_parts = ["用户反馈"]
        
        if "rating" in feedback_data:
            text_parts.append(f"评分: {feedback_data['rating']}")
        
        if "comments" in feedback_data:
            text_parts.append(f"评论: {feedback_data['comments']}")
        
        if "modifications" in feedback_data:
            text_parts.append(f"修改建议: {feedback_data['modifications']}")
        
        return " ".join(text_parts)
    
    def _destination_knowledge_to_text(self, destination: str, knowledge_data: Dict[str, Any]) -> str:
        """将目的地知识转换为文本表示"""
        text_parts = [f"目的地: {destination}"]
        
        if "description" in knowledge_data:
            text_parts.append(f"描述: {knowledge_data['description']}")
        
        if "highlights" in knowledge_data:
            text_parts.append(f"特色: {', '.join(knowledge_data['highlights'])}")
        
        if "best_season" in knowledge_data:
            text_parts.append(f"最佳季节: {knowledge_data['best_season']}")
        
        if "culture" in knowledge_data:
            text_parts.append(f"文化背景: {knowledge_data['culture']}")
        
        return " ".join(text_parts)
    
    def _experience_to_text(self, experience_type: str, experience_data: Dict[str, Any]) -> str:
        """将经验数据转换为文本表示"""
        text_parts = [f"旅行经验: {experience_type}"]
        
        if "title" in experience_data:
            text_parts.append(f"标题: {experience_data['title']}")
        
        if "description" in experience_data:
            text_parts.append(f"描述: {experience_data['description']}")
        
        if "tags" in experience_data:
            text_parts.append(f"标签: {', '.join(experience_data['tags'])}")
        
        if "destination" in experience_data:
            text_parts.append(f"目的地: {experience_data['destination']}")
        
        return " ".join(text_parts)
    
    # ============ 维护方法 ============
    
    def save(self):
        """保存索引和元数据"""
        with self._save_lock:
            if self._save_timer:
                self._save_timer.cancel()
                self._save_timer = None
            self._pending_save = False
        self._save_indexes()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆服务统计信息"""
        return {
            "user_memory_count": self.user_memory_index.ntotal,
            "knowledge_memory_count": self.knowledge_memory_index.ntotal,
            "vector_dimension": self.vector_dim,
            "memory_directory": str(self.memory_dir)
        }


# 创建全局向量记忆服务实例
vector_memory_service = VectorMemoryService()
