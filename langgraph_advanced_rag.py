from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent import MODEL
from langgraph_agent import (
    extract_text,  # AIMessage.content의 str/list[dict] 케이스 처리
)
from metrics import FaithfulnessVerdict  # Self-RAG 자기반성 판정 스키마, 재정의 안 함
from qdrant_store import (
    ALL_COLLECTIONS,
    MultiCollectionStore,
    build_multi_collection_store,
)
from rerank import RerankStore

# 문서 4개짜리 토이 코퍼스라 그 이상 재시도해도 새 정보가 안 나온다 — 최대 1회.
MAX_GENERATION_RETRIES = 1

GENERATE_SYSTEM = (
    "당신은 게임 위키/패치노트 Q&A 어시스턴트입니다. "
    "주어진 [검색된 문서]만 근거로 한국어로 간결하게(2~4문장) 답하세요. "
    "문서에 답이 없으면 지어내지 말고 '문서에서 찾을 수 없습니다'라고 하세요."
)  # agent.py의 SYSTEM과 같은 행동 계약이지만, 여긴 검색이 이미 끝난 뒤라 도구 호출 지시는 뺐다.


class RetrievedDoc(TypedDict):
    text: str
    source: str
    chunk_id: int
    score: float  # 코사인 유사도 (MultiCollectionStore.search()가 리턴)
    # 인덱싱 시점에 미리 붙은 문맥 요약(contextualize.py) — 답변 생성엔 안 쓰고
    # (format_documents()는 이 필드를 참조하지 않음) 관측용으로만 들고 다닌다.
    # 이걸 넣어두면 이 질문의 retrieve_documents 스팬 하나만 열어도 "이 질문으로
    # 검색된 청크에 원래 어떤 context가 붙어있었는지"가 바로 보인다.
    context: str | None


class GraphState(BaseModel):

    question: str  # 원본 질문, 절대 안 바뀜
    route: Literal["chitchat", "wiki"] | None = None
    search_query: str = ""  # store.search()에 실제로 보내는 질의
    collections: list[str] = Field(default_factory=list)  # 멀티문서 라우팅 — 빈 리스트면 전체 검색
    final_answer: str = ""


class WikiState(BaseModel):

    question: str
    search_query: str = ""
    collections: list[str] = Field(default_factory=list)
    documents: list[RetrievedDoc] = Field(default_factory=list)
    generation: str = ""
    faithfulness_verdict: bool = False
    unsupported_claims: list[str] = Field(default_factory=list)
    generation_retry_count: int = 0
    final_answer: str = ""


class QueryAnalysis(BaseModel):
    """질문을 분석해 검색이 필요한지, 필요하다면 어떤 collection(들)에서 검색할지 판단한다.

    collections는 0개 이상 선택 가능 — 확실한 문서 하나면 1개, 근거가 여러 문서에
    걸쳐있을 수 있으면 2개 이상, 정말 불확실하면 빈 리스트(retrieve_documents가
    이걸 "전체 collection 검색"으로 해석한다)."""

    route: Literal["chitchat", "wiki"]
    search_query: str
    collections: list[Literal["bestiary", "patch_notes", "quests", "town_news"]] = Field(
        default_factory=list
    )


def format_documents(documents: list[RetrievedDoc]) -> str:
    """agent.py::run_tool / langgraph_agent.py::search_wiki와 같은 포맷으로 근거 블록을 조립."""
    return "\n\n".join(
        f"[{i}] ({doc['source']} #{doc['chunk_id']}, 유사도 {doc['score']:.3f})\n{doc['text']}"
        for i, doc in enumerate(documents, start=1)
    )


def build_wiki_graph(store: MultiCollectionStore, verbose: bool = True):
    """검색 → 생성 → Self-RAG 자기반성 재시도를 독립된 서브그래프로 컴파일한다."""
    llm = ChatAnthropic(model=MODEL)

    def log(msg: str) -> None:
        if verbose:
            print(f"  {msg}")

    # ── 검색 ──────────────────────────────────────────────────────
    def retrieve_documents(state: WikiState) -> dict:
        # collections가 비었으면(불확실) 4개 전부를, 있으면 그 목록만 검색 대상으로.
        collections = state.collections or ALL_COLLECTIONS
        results = store.search(state.search_query, collections=collections, top_k=3)
        documents: list[RetrievedDoc] = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "score": score,
                "context": chunk.context,
            }
            for chunk, score in results
        ]
        log(
            f"🔎 retrieve(query={state.search_query!r}, "
            f"collections={collections}) → {len(documents)}건"
        )
        return {"documents": documents}

    # ── 생성 (재시도 시 이전에 걸린 미근거 주장을 프롬프트에 명시) ──────────
    def generate_answer(state: WikiState) -> dict:
        user_content = f"[질문]\n{state.question}\n\n[검색된 문서]\n{format_documents(state.documents)}"
        if state.generation_retry_count > 0 and state.unsupported_claims:
            user_content += (
                f"\n\n[주의] 이전 답변에서 문서로 뒷받침되지 않는 주장이 있었다: "
                f"{state.unsupported_claims}. 이번엔 위 문서에 명시된 내용만으로, "
                f"그 주장들을 다시 하지 말고 답하라."
            )
        response = llm.invoke(
            [SystemMessage(content=GENERATE_SYSTEM), HumanMessage(content=user_content)]
        )
        return {"generation": extract_text(response)}

    # ── Self-RAG: 생성된 답변의 자기반성(hallucination 체크) ────────────
    def grade_faithfulness(state: WikiState) -> dict:
        grader = llm.with_structured_output(FaithfulnessVerdict)
        verdict: FaithfulnessVerdict = grader.invoke(
            [
                SystemMessage(
                    content=(
                        "당신은 RAG 시스템 평가자입니다. 주어진 [답변]의 모든 문장이 [참고 문서]의 내용만으로 "
                        "뒷받침되는지 엄격하게 검사하세요. 문서에 없는 내용을 답변이 추가했다면 근거 없는 "
                        "주장으로 판정하세요."
                    )
                ),
                HumanMessage(
                    content=(
                        f"[참고 문서]\n{format_documents(state.documents)}\n\n[답변]\n{state.generation}"
                    )
                ),
            ]
        )
        log(f"🧪 faithful={verdict.is_faithful} (미근거 주장: {verdict.unsupported_claims})")
        update: dict = {
            "faithfulness_verdict": verdict.is_faithful,
            "unsupported_claims": verdict.unsupported_claims,
        }
        if not verdict.is_faithful:
            update["generation_retry_count"] = state.generation_retry_count + 1
        return update

    def accept_answer(state: WikiState) -> dict:
        return {"final_answer": state.generation}

    def give_up(state: WikiState) -> dict:
        return {"final_answer": "문서에서 찾을 수 없습니다."}

    def route_after_faithfulness(state: WikiState) -> str:
        if state.faithfulness_verdict:
            return "accept"
        if state.generation_retry_count <= MAX_GENERATION_RETRIES:
            return "retry"
        return "give_up"

    graph = StateGraph(WikiState)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("grade_faithfulness", grade_faithfulness)
    graph.add_node("accept_answer", accept_answer)
    graph.add_node("give_up", give_up)

    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", "generate_answer")
    graph.add_edge("generate_answer", "grade_faithfulness")
    graph.add_conditional_edges(
        "grade_faithfulness",
        route_after_faithfulness,
        {"accept": "accept_answer", "retry": "generate_answer", "give_up": "give_up"},
    )
    graph.add_edge("accept_answer", END)
    graph.add_edge("give_up", END)

    return graph.compile()


def build_graph(store: MultiCollectionStore, verbose: bool = True):
    """부모 그래프 — Adaptive RAG 라우팅 + wiki_pipeline 서브그래프 배선."""
    llm = ChatAnthropic(model=MODEL)

    def log(msg: str) -> None:
        if verbose:
            print(f"  {msg}")

    # ── Adaptive RAG: 검색이 필요한 질문인지 라우팅 ──────────────────────
    def analyze_query(state: GraphState) -> dict:
        analyzer = llm.with_structured_output(QueryAnalysis)
        result: QueryAnalysis = analyzer.invoke(
            [
                SystemMessage(
                    content=(
                        "게임 위키 Q&A 시스템의 질문 분석기다. 게임 콘텐츠(몬스터/퀘스트/패치/마을 소식)에 "
                        "대한 사실 질문이면 route=wiki로, 인사·잡담이면 route=chitchat으로 판단하라. "
                        "wiki일 때는 위키 검색에 적합한 검색어(search_query)를 만들고, 이 질문의 답이 "
                        "어느 컬렉션에 있을지 collections 배열로 지정한다: bestiary(몬스터 도감), "
                        "patch_notes(패치노트), quests(퀘스트), town_news(마을 소식). "
                        "확실한 문서 하나면 그 1개만, 근거가 여러 문서에 걸쳐 있을 수 있으면 "
                        "관련 있어 보이는 2개 이상을 배열에 넣어라. 정말 판단이 안 되면 빈 배열로 "
                        "둔다(비워두면 4개 컬렉션을 전부 검색한다)."
                    )
                ),
                HumanMessage(content=state.question),
            ]
        )
        log(f"🧭 route={result.route} (query={result.search_query!r}, collections={result.collections})")
        return {
            "route": result.route,
            "search_query": result.search_query or state.question,
            "collections": result.collections,
        }

    def direct_answer(state: GraphState) -> dict:
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "게임 위키 Q&A 어시스턴트다. 지금 질문은 위키와 무관한 인사·잡담이니 "
                        "검색 없이 한국어로 자연스럽게 짧게 답하라."
                    )
                ),
                HumanMessage(content=state.question),
            ]
        )
        return {"final_answer": extract_text(response)}

    def route_after_analysis(state: GraphState) -> str:
        return state.route

    wiki_pipeline = build_wiki_graph(store, verbose)  # 컴파일된 서브그래프

    graph = StateGraph(GraphState)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("wiki_pipeline", wiki_pipeline)  # 컴파일된 그래프를 노드로 그대로 끼워넣는다

    graph.add_edge(START, "analyze_query")
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {"chitchat": "direct_answer", "wiki": "wiki_pipeline"},
    )
    graph.add_edge("direct_answer", END)
    graph.add_edge("wiki_pipeline", END)

    return graph.compile()


def answer(compiled_graph, question: str, session_id: str | None = None) -> str:
    trace_id = Langfuse.create_trace_id()
    langfuse_handler = CallbackHandler(trace_context={"trace_id": trace_id})
    metadata = {"langfuse_trace_name": f"질문: {question[:60]}"}
    if session_id is not None:
        metadata["langfuse_session_id"] = session_id
    initial_state = {"question": question}
    accumulated: dict = dict(initial_state)
    for _namespace, step_state in compiled_graph.stream(
        initial_state,
        stream_mode="updates",
        subgraphs=True,
        config={"callbacks": [langfuse_handler], "metadata": metadata},
    ):
        for update in step_state.values():
            accumulated.update(update)

    if "faithfulness_verdict" in accumulated:
        get_client().create_score(
            trace_id=trace_id,
            name="faithfulness",
            value=1.0 if accumulated["faithfulness_verdict"] else 0.0,
            data_type="BOOLEAN",
            comment=str(accumulated.get("unsupported_claims", [])),
        )

    return accumulated["final_answer"]


def main() -> None:
    load_dotenv()
    # candidate_k=10: MultiCollectionStore.search()는 collections 전체에서 모은 뒤
    # "전체 상위 top_k"만 남기므로(컬렉션별 top_k가 아님), collection=None(4개 전체
    # 검색)일 때 정답이 자기 컬렉션 안에서도 4위 정도로 밀리면 좁은 top_k로는 후보에서
    # 통째로 빠질 수 있다(예: 얼음골렘이 bestiary 안에서 4위). candidate_k를 넉넉히
    # 잡아 리랭커에게 더 넓은 후보를 주고, 최종 3개로 좁히는 건 리랭커가 하게 한다.
    store = RerankStore(build_multi_collection_store(), candidate_k=10)
    compiled_graph = build_graph(store)

    questions = [
        "안녕! 오늘 하루 어땠어?",  # Adaptive RAG: 잡담 → 검색 스킵
        "화염룡 잡는 퀘스트 보상이 뭐야?",  # 정상 경로(1차 검색 성공)
        "저 설산에 사는 애 최근에 patch 있었어?",  # 검색이 부실해도 게이트 없이 generate가 스스로 인정
        "화염룡 잡으면 나오는 전설 무기 이름이 정확히 뭐야?",  # Self-RAG 재시도→give_up
        "이 게임에 PvP 아레나가 있어?",  # 문서에 없음 → generate가 스스로 인정(또는 Self-RAG가 잡아냄)
    ]
    session_id = Langfuse.create_trace_id()
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {answer(compiled_graph, q, session_id=session_id)}")

    store.close()


if __name__ == "__main__":
    main()
