import anthropic
from dotenv import load_dotenv

from store import VectorStore, build_store

load_dotenv()

MODEL = "claude-sonnet-5"


def build_system_prompt() -> str:
    return (
        "당신은 게임 위키/패치노트 Q&A 어시스턴트입니다. "
        "아래 [참고 문서]에 주어진 내용만 근거로 답변하세요. "
        "참고 문서에 답이 없으면 추측하지 말고 '문서에서 찾을 수 없습니다'라고 답하세요. "
        "답변은 한국어로, 간결하게 2~4문장으로 작성하세요."
    )


def build_context_block(results: list[tuple]) -> str:
    """검색된 (청크, 유사도) 목록을 LLM에게 줄 '참고 문서' 텍스트로 조립."""
    parts = []
    for i, (chunk, _score) in enumerate(results, start=1):
        parts.append(f"[문서 {i}] {chunk.text}")
    return "\n\n".join(parts)


def generate_answer(
    store: VectorStore, question: str, top_k: int = 3
) -> tuple[str, list[tuple]]:
    results = store.search(question, top_k=top_k)
    context = build_context_block(results)

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=build_system_prompt(),
        messages=[{
            "role": "user",
            "content": f"[참고 문서]\n{context}\n\n[질문]\n{question}",
        }],
    )
    text = next(b.text for b in message.content if b.type == "text")
    return text, results


def answer(store: VectorStore, question: str, top_k: int = 3) -> None:
    """질문 하나를 처리하고 결과를 화면에 보기 좋게 출력."""
    text, results = generate_answer(store, question, top_k=top_k)

    print(f"\n{'=' * 60}")
    print(f"Q: {question}")
    print('=' * 60)
    print(f"\n{text}\n")

    # 출처는 LLM에게 묻지 않고, 검색 단계에서 이미 확보한 사실을 그대로 표시.
    print("근거 문서 (실제 검색 결과):")
    for chunk, score in results:
        print(f"  - {chunk.source} #{chunk.chunk_id}  (유사도 {score:.3f})")


def main() -> None:
    store = build_store()

    questions = [
        "화염룡은 이번 패치에서 어떻게 바뀌었어?",
        "여관에서 뭐 새로 팔아?",
        "슬라임 잡으면 뭐 나와?",
    ]
    for q in questions:
        answer(store, q)


if __name__ == "__main__":
    main()
