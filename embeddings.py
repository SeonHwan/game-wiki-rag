import numpy as np

# 로컬 임베딩 모델. dense 벡터(차원 1024)만 쓴다.
EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
MAX_LENGTH = 1024

# 실제로 embed()를 호출할 때만 모델을 올린다 — 이 모듈을 import하는 것만으로는
# (예: CI의 구조 테스트처럼 임베딩을 한 번도 안 쓰는 경우) 모델을 받지 않게.
_model = None


def _get_model():
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel

        _model = BGEM3FlagModel(EMBED_MODEL, use_fp16=False)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """여러 텍스트를 한 번에 임베딩. 입력 순서대로 벡터 리스트를 반환."""
    out = _get_model().encode(
        texts,
        batch_size=8,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return np.asarray(out["dense_vecs"], dtype="float32").tolist()


def embed_one(text: str) -> list[float]:
    """텍스트 하나를 임베딩."""
    return embed([text])[0]
