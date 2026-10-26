import numpy as np

from chunking import Chunk, load_and_chunk
from embeddings import embed


def cosine_similarity_batch(query_vec: list[float], matrix: np.ndarray) -> np.ndarray:
    q = np.array(query_vec)
    # 행렬 @ 벡터 = 각 행(청크 벡터)과 q의 내적을 한 번에 계산
    dot_products = matrix @ q
    # 각 청크 벡터의 크기와 질문 벡터의 크기
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
    return dot_products / norms


class VectorStore:
    """청크들을 임베딩해 메모리에 들고, 검색을 제공하는 저장소."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        print(f"{len(chunks)}개 청크를 임베딩하는 중...")
        # 청크 전체를 '한 번의 API 호출'로 임베딩
        vectors = embed([c.text for c in chunks])
        # 리스트의 리스트를 2차원 numpy 배열로.
        self.matrix = np.array(vectors)
        print(f"완료. 행렬 크기: {self.matrix.shape}")

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """질문과 가장 관련 있는 청크 top_k개를 (청크, 유사도점수) 튜플로 반환."""
        query_vec = embed([query])[0]
        scores = cosine_similarity_batch(query_vec, self.matrix)

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.chunks[i], float(scores[i])) for i in top_indices]


def build_store() -> VectorStore:
    """data 폴더를 읽어 청크로 나누고, 임베딩해 VectorStore를 만든다."""
    chunks = load_and_chunk()
    return VectorStore(chunks)
