import json

import anthropic
from dotenv import load_dotenv

from qdrant_store import QdrantStore, build_contextual_store

load_dotenv()

MODEL = "claude-sonnet-5"

SEARCH_TOOL = {
    "name": "search_wiki",
    "description": (
        "게임 위키/패치노트에서 질문과 관련된 문단을 의미 기반으로 검색한다. "
        "몬스터·퀘스트·패치 내역·마을 소식 같은 게임 콘텐츠에 대한 사실을 물으면 "
        "반드시 이 도구로 먼저 근거를 찾은 뒤 답하라. "
        "인사말·잡담처럼 위키와 무관한 입력에는 쓰지 마라."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "검색할 내용 — 질문의 핵심 키워드나 문장",
            },
            "source_filter": {
                "type": "string",
                "description": (
                    "특정 문서 안에서만 찾고 싶을 때 파일명을 지정한다 "
                    "(patch_notes.md / bestiary.md / town_news.md / quests.md). "
                    "어느 문서인지 확실하지 않으면 생략한다(전체 검색)."
                ),
            },
        },
        "required": ["query"],
    },
}

SYSTEM = (
    "당신은 게임 위키/패치노트 Q&A 어시스턴트입니다. "
    "게임 콘텐츠에 대한 사실 질문에는 search_wiki 도구로 근거를 먼저 찾고, "
    "검색된 내용만 근거로 한국어로 간결하게(2~4문장) 답하세요. "
    "검색 결과에 답이 없으면 지어내지 말고 '문서에서 찾을 수 없습니다'라고 하세요. "
    "인사·잡담 등 위키와 무관한 입력에는 도구를 쓰지 말고 자연스럽게 답하세요."
)


def run_tool(store: QdrantStore, tool_name: str, tool_input: dict) -> str:
    if tool_name != "search_wiki":
        return f"알 수 없는 도구입니다: {tool_name}"

    query = tool_input["query"]
    source_filter = tool_input.get("source_filter")  # 모델이 안 줬으면 None(전체 검색)
    results = store.search(query, top_k=3, source_filter=source_filter)

    if not results:
        return "검색 결과가 없습니다."

    # 검색된 청크를 '근거 문서' 텍스트로 조립해 LLM에게 돌려준다.
    parts = []
    for i, (chunk, score) in enumerate(results, start=1):
        parts.append(
            f"[{i}] ({chunk.source} #{chunk.chunk_id}, 유사도 {score:.3f})\n{chunk.text}"
        )
    return "\n\n".join(parts)


def answer(store: QdrantStore, question: str, verbose: bool = True) -> str:
    """질문 하나를 tool-use 루프로 처리하고 최종 답변 텍스트를 반환한다."""
    client = anthropic.Anthropic()

    # 대화 기록. API는 상태가 없으므로(stateless) 매 호출마다 전체를 다시 보낸다.
    messages = [{"role": "user", "content": question}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM,
            tools=[SEARCH_TOOL],  # 이 목록이 곧 "쓸 수 있는 도구"
            messages=messages,
        )

        # 도구를 요청하지 않았다 = 최종 답변을 냈다.
        if response.stop_reason != "tool_use":
            return next((b.text for b in response.content if b.type == "text"), "")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if verbose:
                    args = json.dumps(block.input, ensure_ascii=False)
                    print(f"  🔧 {block.name}({args})")
                result_text = run_tool(store, block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,   # 이 도구 요청 블록의 id와 짝
                    "content": result_text,
                })
        messages.append({"role": "user", "content": tool_results})
        # 루프 계속 → Claude가 이 결과를 보고 다음 행동(또 검색 / 최종 답변)을 정한다.


def main() -> None:
    store = build_contextual_store()
    for q in ["화염룡 잡는 퀘스트 보상이 뭐야?", "슬라임 잡으면 뭐 나와?"]:
        print(f"\nQ: {q}")
        print(f"A: {answer(store, q)}")
    store.close()


if __name__ == "__main__":
    main()
