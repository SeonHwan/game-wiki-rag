from collections import defaultdict
from pathlib import Path

from langfuse import get_client
from qdrant_client import QdrantClient, models

from chunking import DATA_DIR, Chunk, load_and_chunk, load_documents
from contextualize import contextualize_chunks
from embeddings import EMBED_DIM, EMBED_MODEL, embed

QDRANT_PATH = "qdrant_local"

# collection 이름. 하나의 논리적 벡터 묶음(= 하나의 테이블)에 붙이는 이름.
COLLECTION = "game_wiki"

VECTOR_SIZE = EMBED_DIM


class QdrantStore:

    def __init__(
        self,
        chunks: list[Chunk],
        path: str = QDRANT_PATH,
        collection: str = COLLECTION,
    ):
        self.collection = collection
        # 임베디드 모드: path 폴더를 이 클라이언트가 '단독 점유'한다.
        # (같은 폴더를 두 프로세스/두 클라이언트가 동시에 열면 잠금 에러가 난다.
        #  서버 모드에는 없는 임베디드 모드만의 제약.)
        self.client = QdrantClient(path=path)

        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

        print(f"{len(chunks)}개 청크를 임베딩해 Qdrant에 넣는 중...")
        embedding_texts = [c.embedding_text for c in chunks]
        with get_client().start_as_current_observation(
            name="embed_chunks",
            as_type="embedding",
            input=embedding_texts,
            model=EMBED_MODEL,
        ) as obs:
            vectors = embed(embedding_texts)
            obs.update(output=f"{len(vectors)}개 벡터, 차원={len(vectors[0]) if vectors else 0}")
        points = [
            models.PointStruct(
                id=i,
                vector=vec,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "context": chunk.context,
                },
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=collection, points=points)
        print(f"완료. collection '{collection}'에 {len(points)}개 point 저장.")

    def search(
        self,
        query: str,
        top_k: int = 3,
        source_filter: str | None = None,
    ) -> list[tuple[Chunk, float]]:
        query_vec = embed([query])[0]

        query_filter = None
        if source_filter is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source_filter),
                    )
                ]
            )

        # query_points: 최신 검색 API. query에 질문 벡터를 주면 가장 가까운 point들을 반환.
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vec,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,  # payload(원문+메타)도 함께 돌려받는다.
        )

        results: list[tuple[Chunk, float]] = []
        for point in response.points:
            p = point.payload
            chunk = Chunk(
                text=p["text"],
                source=p["source"],
                chunk_id=p["chunk_id"],
                context=p.get("context"),
            )
            # point.score = 코사인 유사도 (v1의 score와 같은 의미).
            results.append((chunk, float(point.score)))
        return results

    def close(self) -> None:
        self.client.close()


def build_qdrant_store() -> QdrantStore:
    chunks = load_and_chunk()
    return QdrantStore(chunks)


def build_contextual_store(
    path: str = "qdrant_local_contextual",
    collection: str = "game_wiki_ctx",
) -> QdrantStore:
    chunks = load_and_chunk()
    documents = load_documents()
    contextual_chunks = contextualize_chunks(chunks, documents)
    return QdrantStore(contextual_chunks, path=path, collection=collection)


# 멀티문서 라우팅용 collection 이름들 — data/*.md 파일명에서 확장자만 뗀 것.
# 하드코딩 안 하고 실제 파일 목록에서 계산 → 문서가 추가/삭제돼도 자동으로 맞음.
ALL_COLLECTIONS = sorted(p.stem for p in DATA_DIR.glob("*.md"))


def _collection_name(source: str) -> str:
    """청크의 출처 파일명(예: "patch_notes.md")을 collection 이름("patch_notes")으로."""
    return Path(source).stem


class MultiCollectionStore:

    def __init__(self, chunks: list[Chunk], path: str = "qdrant_local_multi"):
        self.client = QdrantClient(path=path)

        by_source: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in chunks:
            by_source[chunk.source].append(chunk)

        for source, group in sorted(by_source.items()):
            name = _collection_name(source)

            if self.client.collection_exists(name):
                self.client.delete_collection(name)
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

            embedding_texts = [c.embedding_text for c in group]
            with get_client().start_as_current_observation(
                name=f"embed_chunks:{name}",
                as_type="embedding",
                input=embedding_texts,
                model=EMBED_MODEL,
            ) as obs:
                vectors = embed(embedding_texts)
                obs.update(output=f"{len(vectors)}개 벡터, 차원={len(vectors[0]) if vectors else 0}")

            points = [
                models.PointStruct(
                    id=i,
                    vector=vec,
                    payload={
                        "text": chunk.text,
                        "source": chunk.source,
                        "chunk_id": chunk.chunk_id,
                        "context": chunk.context,
                    },
                )
                for i, (chunk, vec) in enumerate(zip(group, vectors))
            ]
            self.client.upsert(collection_name=name, points=points)
            print(f"collection '{name}'에 {len(points)}개 point 저장.")

    def search(
        self,
        query: str,
        collections: list[str],
        top_k: int = 3,
    ) -> list[tuple[Chunk, float]]:
        query_vec = embed([query])[0]

        all_points = []
        for name in collections:
            response = self.client.query_points(
                collection_name=name,
                query=query_vec,
                limit=top_k,
                with_payload=True,
            )
            all_points.extend(response.points)

        # 여러 collection에서 모인 결과를 점수 기준으로 정렬 — 같은 임베딩 모델/
        # 공간을 쓰므로 컬렉션이 달라도 코사인 유사도끼리 직접 비교 가능하다.
        # (자르지 않고 전부 반환 — 위 docstring 참고.)
        all_points.sort(key=lambda point: point.score, reverse=True)

        results: list[tuple[Chunk, float]] = []
        for point in all_points:
            p = point.payload
            chunk = Chunk(
                text=p["text"],
                source=p["source"],
                chunk_id=p["chunk_id"],
                context=p.get("context"),
            )
            results.append((chunk, float(point.score)))
        return results

    def close(self) -> None:
        self.client.close()


def build_multi_collection_store() -> MultiCollectionStore:
    chunks = load_and_chunk()
    documents = load_documents()
    contextual_chunks = contextualize_chunks(chunks, documents)
    return MultiCollectionStore(contextual_chunks)
