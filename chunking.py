from pathlib import Path

from pydantic import BaseModel

DATA_DIR = Path(__file__).parent / "data"


class Chunk(BaseModel):
    """검색 단위가 되는 문서 조각."""
    text: str        # 조각 내용
    source: str      # 출처 파일명 (예: "patch_notes.md") — 인용에 사용
    chunk_id: int    # 그 파일 안에서 몇 번째 조각인지
    context: str | None = None

    @property
    def embedding_text(self) -> str:
        """실제로 임베딩할 텍스트. context가 있으면 앞에 붙여서 문맥을 보강한다."""
        if self.context is None:
            return self.text
        return f"{self.context}\n\n{self.text}"


def load_and_chunk() -> list[Chunk]:
    """data 폴더의 모든 .md 파일을 읽어 문단 단위 청크 리스트로 반환."""
    chunks: list[Chunk] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        # 빈 줄(\n\n) 기준으로 문단 분리. 공백뿐인 조각은 버림.
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            chunks.append(Chunk(text=para, source=path.name, chunk_id=i))
    return chunks


def load_documents() -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(DATA_DIR.glob("*.md"))
    }
