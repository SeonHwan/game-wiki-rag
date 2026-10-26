import asyncio
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from agent import MODEL, SYSTEM  # phase1의 모델·시스템 프롬프트 재사용
from langgraph_agent import (  # 그래프 State와 텍스트 추출은 그대로 재사용
    State,
    extract_text,
)


def build_graph(tools: list):
    llm_with_tools = ChatAnthropic(model=MODEL).bind_tools(tools)

    async def agent_node(state: State) -> dict:
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))  # MCP 도구라도 ToolNode 배선은 동일

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


async def answer(compiled_graph, question: str, verbose: bool = True) -> str:
    state: State = {"messages": [SystemMessage(content=SYSTEM), HumanMessage(content=question)]}

    last: AIMessage | None = None
    async for step_state in compiled_graph.astream(state, stream_mode="updates"):
        for update in step_state.values():
            for msg in update["messages"]:
                last = msg
                if verbose and isinstance(msg, AIMessage) and msg.tool_calls:
                    for call in msg.tool_calls:
                        print(f"  🔧 {call['name']}({call['args']}) → MCP 서버로 실행 위임")

    return extract_text(last)


async def main() -> None:
    load_dotenv()
    server = StdioServerParameters(command=sys.executable, args=["mcp_server.py"])

    # mcp_agent.py와 완전히 동일한 방식으로 서버를 하위 프로세스로 띄워 붙는다.
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)
            print(f"MCP 서버에서 발견한 도구: {[t.name for t in tools]}\n")

            compiled_graph = build_graph(tools)

            # ② mcp_agent.py와 동일한 두 질문으로 비교.
            for q in ["화염룡 잡는 퀘스트 보상이 뭐야?", "안녕! 너는 무슨 일을 하는 봇이야?"]:
                print("=" * 70)
                print(f"Q: {q}")
                result = await answer(compiled_graph, q)
                print(f"\nA: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())
