import anthropic
from dotenv import load_dotenv
from langfuse import observe

from chunking import Chunk

load_dotenv()

MODEL = "claude-sonnet-5"

SYSTEM = (
    "당신은 검색 품질 개선을 위해 문서 조각(청크)에 짧은 문맥을 붙이는 어시스턴트입니다. "
    "주어진 [전체 문서]를 참고해서, [대상 청크]가 문서의 어느 내용을 가리키는지, "
    "그 안의 대명사나 생략된 주어가 무엇을 뜻하는지 보충하는 한두 문장을 한국어로 쓰세요. "
    "다른 설명 없이 그 문맥 문장만 답하세요."
)


@observe(as_type="generation", name="generate_context")
def generate_context(chunk_text: str, doc_text: str) -> str:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=150,
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"[전체 문서]\n{doc_text}\n\n"
                f"[대상 청크]\n{chunk_text}"
            ),
        }],
    )
    return next(b.text for b in message.content if b.type == "text").strip()


@observe(name="contextualize_chunks")
def contextualize_chunks(
    chunks: list[Chunk],
    documents: dict[str, str],
) -> list[Chunk]:
    result: list[Chunk] = []
    for i, chunk in enumerate(chunks, start=1):
        context = generate_context(chunk.text, documents[chunk.source])
        result.append(chunk.model_copy(update={"context": context}))
        print(f"  [{i}/{len(chunks)}] {chunk.source}#{chunk.chunk_id} 문맥 생성 완료")
    return result
