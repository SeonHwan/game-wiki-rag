import json
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent import MODEL, SYSTEM  # phase1과 같은 모델·시스템 프롬프트 재사용
from qdrant_store import QdrantStore, build_qdrant_store


class State(TypedDict):
    messages: Annotated[list, add_messages]


def make_search_tool(store: QdrantStore):

    @tool
    def search_wiki(query: str, source_filter: str | None = None) -> str:
        """게임 위키/패치노트에서 질문과 관련된 문단을 의미 기반으로 검색한다.

        몬스터·퀘스트·패치 내역·마을 소식 같은 게임 콘텐츠에 대한 사실을 물으면
        반드시 이 도구로 먼저 근거를 찾은 뒤 답하라. 인사말·잡담처럼 위키와
        무관한 입력에는 쓰지 마라.

        Args:
            query: 검색할 내용 — 질문의 핵심 키워드나 문장.
            source_filter: 특정 문서로 좁힐 때 파일명을 지정
                (patch_notes.md / bestiary.md / town_news.md / quests.md).
                어느 문서인지 확실하지 않으면 생략한다(전체 검색).
        """
        results = store.search(query, top_k=3, source_filter=source_filter)
        if not results:
            return "검색 결과가 없습니다."
        parts = []
        for i, (chunk, score) in enumerate(results, start=1):
            parts.append(
                f"[{i}] ({chunk.source} #{chunk.chunk_id}, 유사도 {score:.3f})\n{chunk.text}"
            )
        return "\n\n".join(parts)

    return search_wiki


def build_graph(tools: list):
    """State/노드/엣지를 배선해서 컴파일된 그래프를 돌려준다."""
    llm_with_tools = ChatAnthropic(model=MODEL).bind_tools(tools)

    def agent_node(state: State) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}  # 이 한 개만 리턴하면 add_messages가 누적한다

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))  # agent.py의 run_tool()이 하던 실행을 대신

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")  # 도구 실행 후엔 다시 agent로 — agent.py의 while True

    return graph.compile()


def extract_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
    )


def answer(compiled_graph, question: str, verbose: bool = True) -> str:
    """질문 하나를 그래프에 흘려보내고 최종 답변 텍스트를 반환한다."""
    state: State = {"messages": [SystemMessage(content=SYSTEM), HumanMessage(content=question)]}

    last: AIMessage | None = None
    for step_state in compiled_graph.stream(state, stream_mode="updates"):
        for update in step_state.values():
            for msg in update["messages"]:
                last = msg
                if verbose and isinstance(msg, AIMessage) and msg.tool_calls:
                    for call in msg.tool_calls:
                        args = json.dumps(call["args"], ensure_ascii=False)
                        print(f"  🔧 {call['name']}({args})")

    return extract_text(last)


def main() -> None:
    load_dotenv()
    store = build_qdrant_store()
    tools = [make_search_tool(store)]
    compiled_graph = build_graph(tools)

    for q in ["화염룡 잡는 퀘스트 보상이 뭐야?", "슬라임 잡으면 뭐 나와?"]:
        print(f"\nQ: {q}")
        print(f"A: {answer(compiled_graph, q)}")

    store.close()


if __name__ == "__main__":
    main()
