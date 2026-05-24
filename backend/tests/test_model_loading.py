"""
Smoke test for the vector memory embedding service.

The project uses DashScope's OpenAI-compatible embeddings endpoint by default:
text-embedding-v4, 1024 dimensions.
"""

from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def test_embedding_service() -> bool:
    from app.config import settings
    from app.services.vector_memory_service import VectorMemoryService

    print("=" * 80)
    print("Vector memory embedding smoke test")
    print("=" * 80)
    print(f"Provider: {settings.EMBEDDING_PROVIDER}")
    print(f"Model: {settings.EMBEDDING_MODEL}")
    print(f"Base URL: {settings.EMBEDDING_BASE_URL or settings.LLM_BASE_URL}")
    print(f"Vector dim: {settings.VECTOR_DIM}")

    memory_service = VectorMemoryService()
    vector = memory_service._text_to_vector("杭州西湖适合两日休闲旅行")

    print(f"Generated vector dimension: {len(vector)}")
    print(f"Vector norm: {(vector ** 2).sum() ** 0.5:.4f}")

    assert len(vector) == settings.VECTOR_DIM
    assert vector.any(), "embedding vector is all zeros; check API key/base URL/model"
    print("OK")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if test_embedding_service() else 1)
