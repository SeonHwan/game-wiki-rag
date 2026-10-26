import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

MODEL = "claude-sonnet-5"


class Triple(BaseModel):
    """지식그래프의 최소 단위: (주어) --[관계]--> (목적어)."""
    subject: str
    relation: str
    object: str
    source: str  # 어느 문서에서 뽑혔는지 — 인용/디버깅용


class _ExtractedTriple(BaseModel):
    subject: str
    relation: str
    object: str


class _ExtractionResult(BaseModel):
    triples: list[_ExtractedTriple]


EXTRACT_TOOL = {
    "name": "report_triples",
    "description": "문서에서 뽑은 (주어, 관계, 목적어) 트리플 목록을 반환한다.",
    "input_schema": _ExtractionResult.model_json_schema(),
}

SYSTEM = (
    "당신은 게임 위키 문서에서 지식그래프용 트리플을 추출하는 어시스턴트입니다. "
    "주어진 [문서]를 읽고, 그 안에 명시된 사실을 (주어, 관계, 목적어) 형태로 전부 뽑으세요.\n\n"
    "규칙:\n"
    "1. 주어와 목적어는 둘 다 반드시 '개체명 하나'여야 합니다. 서술적 구절, 조건문, "
    "여러 개체가 뒤섞인 문구를 그대로 목적어에 넣으면 안 됩니다.\n"
    "2. 한 문장에 사실이 여러 개 얽혀 있으면 여러 개의 트리플로 쪼개세요. 예를 들어 "
    "'탑의 시험은 탑 최상층의 바람의 정령을 처치하면 완료된다'는 이렇게 두 개로 나눕니다:\n"
    "   맞는 예: (탑의 시험, 대상 몬스터, 바람의 정령) / (바람의 정령, 서식지, 탑 최상층)\n"
    "   틀린 예: (탑의 시험, 완료조건, 탑 최상층의 바람의 정령 처치) "
    "<- 목적어가 통짜 구절이라 잘못됨\n"
    "3. 개체명은 문서 전체에서 반복되는 핵심 명사만 쓰고, 괄호 안 별칭·영문 표기는 빼세요 "
    "(예: '화염룡(Fire Dragon)'이 아니라 '화염룡').\n"
    "4. 관계는 짧은 명사/동사구로 쓰세요(예: 서식지, 보상, 드롭, 약점, 대상 몬스터)."
)


def extract_triples(doc_name: str, doc_text: str) -> list[Triple]:
    """문서 하나에서 트리플을 전부 뽑아 Triple 리스트로 반환(source=doc_name)."""
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        # 트리플 하나당 필드 3개(주어/관계/목적어)를 매번 다 쓰다 보니 토큰을 꽤 먹는다.
        # 문서 하나(400~700자)당 트리플 추출에 1000토큰 넘게 쓰는 경우가 있었고,
        # max_tokens=1024였을 때 bestiary.md/patch_notes.md가 stop_reason="max_tokens"로
        # 잘려서 tool input이 빈 dict로 옴(JSON을 못 끝내서). 넉넉히 잡아야 안전하다.
        max_tokens=4096,
        # temperature=0은 쓰지 않는다. claude-sonnet-5는 temperature 파라미터를 받지 않아
        # "temperature is deprecated for this model"(400 에러)이 난다. 그래서 같은 문서에서도
        # 추출 결과가 실행마다 달라질 수 있다.
        system=SYSTEM,
        messages=[{"role": "user", "content": f"[문서: {doc_name}]\n{doc_text}"}],
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "report_triples"},
    )
    tool_block = next(b for b in message.content if b.type == "tool_use")
    if message.stop_reason == "max_tokens":
        raise RuntimeError(
            f"{doc_name}: 트리플 추출이 max_tokens(4096)에서도 잘림 — 문서를 나눠서 "
            f"추출하거나 max_tokens를 더 늘려야 함."
        )
    result = _ExtractionResult.model_validate(tool_block.input)
    return [
        Triple(subject=t.subject, relation=t.relation, object=t.object, source=doc_name)
        for t in result.triples
    ]


def extract_all_triples(documents: dict[str, str]) -> list[Triple]:
    """load_documents()가 준 {파일명: 전체텍스트} 전부에서 트리플을 뽑아 하나로 합친다."""
    all_triples: list[Triple] = []
    for doc_name, doc_text in documents.items():
        triples = extract_triples(doc_name, doc_text)
        print(f"  {doc_name}: {len(triples)}개 트리플 추출")
        all_triples.extend(triples)
    return all_triples
