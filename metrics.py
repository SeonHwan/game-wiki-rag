import anthropic
from pydantic import BaseModel

from chunking import Chunk
from embeddings import embed_one

MODEL = "claude-sonnet-5"


# ── 1. Context Precision@k ────────────────────────────────────────────
def context_precision_at_k(retrieved: list[Chunk], expected_source: str) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for c in retrieved if c.source == expected_source)
    return hits / len(retrieved)


# ── 2. Context Recall ─────────────────────────────────────────────────
def context_recall(retrieved: list[Chunk], expected_source: str) -> float:
    found = any(c.source == expected_source for c in retrieved)
    return 1.0 if found else 0.0


def precision_at_1(retrieved: list[Chunk], expected_source: str) -> float:
    if not retrieved:
        return 0.0
    return 1.0 if retrieved[0].source == expected_source else 0.0


def mrr(retrieved: list[Chunk], expected_source: str) -> float:
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk.source == expected_source:
            return 1.0 / rank
    return 0.0


# ── 3. Faithfulness (LLM 심판) ────────────────────────────────────────
class FaithfulnessVerdict(BaseModel):
    # tool 의 input_schema로 변환해 사용
    is_faithful: bool
    unsupported_claims: list[str]  # 문서에서 근거를 못 찾은 주장들 (없으면 빈 리스트)


FAITHFULNESS_TOOL = {
    "name": "report_faithfulness",
    "description": "답변이 참고 문서에 근거하는지 판정한 결과를 반환한다.",
    "input_schema": FaithfulnessVerdict.model_json_schema(),
}


def faithfulness(answer_text: str, context: str) -> float:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=(
            "당신은 RAG 시스템 평가자입니다. 주어진 [답변]의 모든 문장이 "
            "[참고 문서]의 내용만으로 뒷받침되는지 엄격하게 검사하세요. "
            "문서에 없는 내용을 답변이 추가했다면 근거 없는 주장으로 판정하세요."
        ),
        messages=[{
            "role": "user",
            "content": f"[참고 문서]\n{context}\n\n[답변]\n{answer_text}",
        }],
        tools=[FAITHFULNESS_TOOL],
        tool_choice={"type": "tool", "name": "report_faithfulness"},
    )
    tool_block = next(b for b in message.content if b.type == "tool_use")
    verdict = FaithfulnessVerdict.model_validate(tool_block.input)
    return 1.0 if verdict.is_faithful else 0.0


# ── 4. Answer Relevancy (임베딩 유사도) ───────────────────────────────
def answer_relevancy(question: str, answer_text: str) -> float:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=128,
        system=(
            "다음 [답변]만 보고, 이 답변이 나오게 된 질문이 무엇일지 한 문장으로 "
            "추측해 질문 형태로만 답하세요. 다른 설명은 붙이지 마세요."
        ),
        messages=[{"role": "user", "content": f"[답변]\n{answer_text}"}],
    )
    reconstructed_question = next(
        b.text for b in message.content if b.type == "text"
    ).strip()

    # embeddings.py의 코사인 유사도 아이디어를 재사용.
    import numpy as np
    v1 = np.array(embed_one(question))
    v2 = np.array(embed_one(reconstructed_question))
    return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
