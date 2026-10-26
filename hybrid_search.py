import re

import numpy as np
from rank_bm25 import BM25Okapi

from chunking import Chunk, load_and_chunk
from embeddings import embed


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w]", "", text)  # 공백·문장부호 제거
    if len(cleaned) < 2:
        return [cleaned] if cleaned else []
    return [cleaned[i:i + 2] for i in range(len(cleaned) - 1)]


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:   # 전부 같은 값이면 0으로 나누기 방지
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class HybridStore:

    def __init__(self, chunks: list[Chunk], alpha: float = 0.5):
        """alpha: 벡터 점수 비중. 0=순수 키워드, 1=순수 벡터, 0.5=반반."""
        self.chunks = chunks
        self.alpha = alpha

        print(f"{len(chunks)}개 청크를 임베딩하는 중...")
        vectors = embed([c.text for c in chunks])
        self.matrix = np.array(vectors)

        print("BM25 키워드 인덱스를 만드는 중...")
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        # ① 벡터 유사도 — store.py의 cosine_similarity_batch와 동일한 계산
        query_vec = np.array(embed([query])[0])
        dot = self.matrix @ query_vec
        norms = np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(query_vec)
        vector_scores = dot / norms

        # ② BM25 키워드 점수
        bm25_scores = np.array(self.bm25.get_scores(_tokenize(query)))

        # ③ 스케일이 다른 두 점수를 0~1로 맞추고 가중합
        combined = (
            self.alpha * _minmax(vector_scores)
            + (1 - self.alpha) * _minmax(bm25_scores)
        )

        top_indices = np.argsort(combined)[::-1][:top_k]
        return [(self.chunks[i], float(combined[i])) for i in top_indices]


def build_hybrid_store(alpha: float = 0.5) -> HybridStore:
    """data 폴더를 읽어 청크로 나누고, HybridStore를 만든다."""
    chunks = load_and_chunk()
    return HybridStore(chunks, alpha=alpha)
