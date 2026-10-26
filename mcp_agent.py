import asyncio
import json

from anthropic import AsyncAnthropic
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from agent import MODEL, SYSTEM


def mcp_tools_to_anthropic(mcp_tools) -> list[dict]:
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema,
        }
        for t in mcp_tools
    ]


async def call_mcp_tool(session: ClientSession, name: str, arguments: dict) -> str:
    """MCP 세션으로 도구를 실행하고, 결과 content에서 text만 뽑아 반환."""
    result = await session.call_tool(name, arguments)
    return "\n".join(b.text for b in result.content if b.type == "text")


async def answer(llm, session, tools, question: str, verbose: bool = True) -> str:
    """phase1의 tool-use 루프 — 단, 도구 실행을 MCP 서버로 위임한다(async)."""
    messages = [{"role": "user", "content": question}]

    while True:
        response = await llm.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM,
            tools=tools,       # ← MCP에서 발견한 도구 목록
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return next((b.text for b in response.content if b.type == "text"), "")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if verbose:
                    args = json.dumps(block.input, ensure_ascii=False)
                    print(f"  🔧 {block.name}({args}) → MCP 서버로 실행 위임")
                # phase1은 여기서 로컬 run_tool()을 불렀지만, 이제 MCP로 넘긴다.
                text = await call_mcp_tool(session, block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": text,
                })
        messages.append({"role": "user", "content": tool_results})


async def main() -> None:
    llm = AsyncAnthropic()
    server = StdioServerParameters(command="python", args=["mcp_server.py"])

    # MCP 서버를 하위 프로세스로 띄워 붙는다
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 도구를 'MCP 서버에서 발견'해 Anthropic 형식으로 변환
            discovered = (await session.list_tools()).tools
            tools = mcp_tools_to_anthropic(discovered)
            print(f"MCP 서버에서 발견한 도구: {[t['name'] for t in tools]}\n")

            # 검색 필요 질문 vs 잡담.
            for q in ["화염룡 잡는 퀘스트 보상이 뭐야?", "안녕! 너는 무슨 일을 하는 봇이야?"]:
                print("=" * 70)
                print(f"Q: {q}")
                result = await answer(llm, session, tools, q)
                print(f"\nA: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())
