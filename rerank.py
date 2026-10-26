import anthropic
from pydantic import BaseModel

from chunking import Chunk

MODEL = "claude-sonnet-5"


class RerankVerdict(BaseModel):
    ranking: list[int]  # 후보 인덱스(0부터)를 관련도 높은 순으로 나열


RERANK_TOOL = {
    "name": "report_ranking",
    "description": "후보 문서들을 질문에 대한 관련도가 높은 순서로 재정렬한 결과를 반환한다.",
    "input_schema": RerankVerdict.model_json_schema(),
}


def rerank(
    question: str, candidates: list[tuple[Chunk, float]]
) -> list[tuple[Chunk, float]]:
    listing = "\n\n".join(
        f"[{i}] {chunk.text}" for i, (chunk, _score) in enumerate(candidates)
    )

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=(
            "당신은 검색 결과 재정렬 평가자입니다. 아래 [질문]에 가장 정확하고 "
            "직접적으로 답하는 순서로 [후보 문서]들의 인덱스를 나열하세요. "
            "단순히 주제가 겹치는 것보다, 질문이 실제로 요구하는 정보(이유, "
            "방법, 수치 등)를 담고 있는 문서를 더 높은 순위로 두세요."
        ),
        messages=[{
            "role": "user",
            "content": f"[질문]\n{question}\n\n[후보 문서]\n{listing}",
        }],
        tools=[RERANK_TOOL],
        tool_choice={"type": "tool", "name": "report_ranking"},
    )
    tool_block = next(b for b in message.content if b.type == "tool_use")
    verdict = RerankVerdict.model_validate(tool_block.input)

    # LLM이 인덱스를 빠뜨리거나 중복 낼 수도 있으니 방어적으로 처리:
    # 유효하고 처음 나온 인덱스만 순서대로 채택하고, 빠진 인덱스는 원래 순서로 뒤에 붙인다.
    seen: set[int] = set()
    ordered: list[tuple[Chunk, float]] = []
    for idx in verdict.ranking:
        if 0 <= idx < len(candidates) and idx not in seen:
            ordered.append(candidates[idx])
            seen.add(idx)
    for i, c in enumerate(candidates):
        if i not in seen:
            ordered.append(c)
    return ordered


class RerankStore:

    def __init__(self, base_store, candidate_k: int = 6):
        self.base_store = base_store
        self.candidate_k = candidate_k

    def search(self, query: str, top_k: int = 3, **kwargs) -> list[tuple[Chunk, float]]:
        # **kwargs를 그대로 base_store에 전달 — MultiCollectionStore.search()의
        # collections= 같은 새 인자가 생겨도 RerankStore가 매번 안 고쳐져도 됨.
        candidates = self.base_store.search(query, top_k=self.candidate_k, **kwargs)
        reranked = rerank(query, candidates)
        return reranked[:top_k]

    def close(self) -> None:
        self.base_store.close()
