import atexit
import contextlib
import sys

from mcp.server.fastmcp import FastMCP

from qdrant_store import build_qdrant_store

# stdio 서버에서 stdout은 '프로토콜 채널'이다 — 여기에 잡다한 print가 섞이면
# JSON-RPC 스트림이 깨진다. 서버 구성 동안의 표준출력을 stderr로 돌려서 stdio 채널을 깨지 않게 한다.
with contextlib.redirect_stdout(sys.stderr):
    _store = build_qdrant_store()

atexit.register(_store.close)

mcp = FastMCP("game-wiki")


@mcp.tool()
def search_wiki(query: str, source_filter: str | None = None) -> str:
    """게임 위키/패치노트에서 질문과 관련된 문단을 의미 기반으로 검색한다.

    몬스터·퀘스트·패치 내역·마을 소식 같은 게임 콘텐츠에 대한 사실의 근거를 찾을 때 쓴다.

    Args:
        query: 검색할 내용 — 질문의 핵심 키워드나 문장.
        source_filter: 특정 문서로 좁힐 때 파일명을 지정
            (patch_notes.md / bestiary.md / town_news.md / quests.md).
            어느 문서인지 확실하지 않으면 생략한다(전체 검색).
    """
    results = _store.search(query, top_k=3, source_filter=source_filter)
    if not results:
        return "검색 결과가 없습니다."
    parts = []
    for i, (chunk, score) in enumerate(results, start=1):
        parts.append(
            f"[{i}] ({chunk.source} #{chunk.chunk_id}, 유사도 {score:.3f})\n{chunk.text}"
        )
    return "\n\n".join(parts)


if __name__ == "__main__":
    mcp.run(transport="stdio")
