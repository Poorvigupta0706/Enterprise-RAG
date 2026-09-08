from __future__ import annotations

from dataclasses import dataclass

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from config import Settings, settings
from fusion import reciprocal_rank_fusion
from ingest import load_chunks
from type import Chunk, RetrievedDoc


def _as_list(vec) -> list:
    if hasattr(vec, "tolist"):
        return vec.tolist()
    return list(vec)


@dataclass
class HybridIndex:
    client: QdrantClient
    dense: TextEmbedding
    sparse: SparseTextEmbedding
    collection: str
    chunks: list[Chunk]


def connect_qdrant(cfg: Settings | None = None) -> QdrantClient:
    cfg = cfg or settings

    print("QDRANT URL =", cfg.qdrant_url)

    if cfg.qdrant_url:
        print("Using Docker Qdrant")
        return QdrantClient(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key or None
            
        )

    print("Using In-Memory Qdrant")
    return QdrantClient(":memory:")


def build_index(
    client: QdrantClient | None = None,
    cfg: Settings | None = None,
) -> HybridIndex:
    cfg = cfg or settings
    client = client or connect_qdrant(cfg)
    dense = TextEmbedding(model_name=cfg.dense_model)
    sparse = SparseTextEmbedding(model_name=cfg.sparse_model)
    chunks = load_chunks()

    if client.collection_exists(cfg.qdrant_collection):
        client.delete_collection(cfg.qdrant_collection)

    client.create_collection(
        collection_name=cfg.qdrant_collection,
        vectors_config={
            "dense": models.VectorParams(size=cfg.dense_dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    dense_vecs = list(dense.embed(c.text for c in chunks))
    sparse_vecs = list(sparse.embed(c.text for c in chunks))
    points: list[models.PointStruct] = []
    for i, chunk in enumerate(chunks):
        sv = sparse_vecs[i]
        points.append(
            models.PointStruct(
                id=i,
                payload={
                    "chunk_id": chunk.id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "text": chunk.text,
                },
                vector={
                    "dense": _as_list(dense_vecs[i]),
                    "bm25": models.SparseVector(
                        indices=_as_list(sv.indices),
                        values=_as_list(sv.values),
                    ),
                },
            )
        )
    client.upsert(collection_name=cfg.qdrant_collection, points=points)
    return HybridIndex(
        client=client,
        dense=dense,
        sparse=sparse,
        collection=cfg.qdrant_collection,
        chunks=chunks,
    )


def _to_docs(points: list[models.ScoredPoint], rank_field: str) -> list[RetrievedDoc]:
    docs: list[RetrievedDoc] = []
    for rank, point in enumerate(points, start=1):
        payload = point.payload or {}
        kwargs: dict = {}
        kwargs[rank_field] = rank
        docs.append(
            RetrievedDoc(
                chunk_id=str(payload.get("chunk_id", point.id)),
                title=str(payload.get("title", "")),
                source=str(payload.get("source", "")),
                text=str(payload.get("text", "")),
                score=float(point.score or 0.0),
                **kwargs,
            )
        )
    return docs


def dense_search(index: HybridIndex, query: str, limit: int) -> list[RetrievedDoc]:
    vector = _as_list(next(index.dense.embed([query])))
    res = index.client.query_points(
        collection_name=index.collection,
        query=vector,
        using="dense",
        limit=limit,
        with_payload=True,
    )
    return _to_docs(res.points, "dense_rank")


def bm25_search(index: HybridIndex, query: str, limit: int) -> list[RetrievedDoc]:
    sv = next(index.sparse.query_embed([query]))
    res = index.client.query_points(
        collection_name=index.collection,
        query=models.SparseVector(indices=_as_list(sv.indices), values=_as_list(sv.values)),
        using="bm25",
        limit=limit,
        with_payload=True,
    )
    return _to_docs(res.points, "bm25_rank")


def hybrid_search(
    index: HybridIndex,
    query: str,
    limit: int | None = None,
    cfg: Settings | None = None,
) -> list[RetrievedDoc]:
    cfg = cfg or settings
    limit = limit or cfg.hybrid_candidates
    dense_hits = dense_search(index, query, limit)
    sparse_hits = bm25_search(index, query, limit)
    return reciprocal_rank_fusion([dense_hits, sparse_hits])[:limit]
