# game-wiki-rag

가상의 게임 위키(몬스터 도감, 패치노트, 퀘스트, 마을 소식)를 대상으로 한 RAG 질의응답 시스템입니다.
검색 → 답변 생성의 기본 파이프라인에서 시작해, 검색 품질을 숫자로 재면서 하이브리드 검색, 리랭커, Contextual Retrieval, GraphRAG를 더하고,
이를 tool-use 에이전트, MCP 서버, LangGraph 기반 Self-RAG/Adaptive RAG 파이프라인으로 확장한 구현입니다.

데이터는 LLM이 사전 지식으로 답할 수 없도록 직접 만든 가상 게임 문서 4개(`data/*.md`)입니다.

## AI Agent 단계별 프로젝트

AI Agent를 단계별로 만들어 가는 5개 프로젝트 중 **1번째**입니다. 앞 프로젝트에서 남은 한계가 다음 프로젝트를 시작한 이유입니다.

| 순서 | 프로젝트 | 다룬 것 | 남은 한계 → 다음 단계 |
|---|---|---|---|
| 1 | **[game-wiki-rag](https://github.com/SeonHwan/game-wiki-rag)** | 가상 게임 위키 4문서로 RAG 기본 파이프라인, 하이브리드·리랭커·Contextual Retrieval·GraphRAG, tool-use 에이전트, MCP, LangGraph 기반 Adaptive/Self-RAG | 데이터와 평가 질문이 너무 작아(문서 4개, 질문 12개) 오케스트레이션, 답변 품질 개선, 여러 RAG 기법의 효과를 제대로 확인하기 어려움 → 실제 방대한 데이터로 다시 |
| 2 | [dart-filing-agent](https://github.com/SeonHwan/dart-filing-agent) | DART 사업보고서 5개사 15건(수만 개 청크): 표 행 단위 청킹, bge-m3 dense+sparse 하이브리드, 리랭커 측정, XBRL 값 조회 스토어 | 문서가 전문용어 중심이라 답변이 나아졌는지 확인하려면 용어부터 이해해야 해서 학습 진도가 더딤 → 일반 상식에 가까운 데이터로 |
| 3 | [encykorea-agent](https://github.com/SeonHwan/encykorea-agent) | 한국민족문화대백과사전 15,836건(청크 75,122개): hybrid 검색, 답변 생성 방식 비교, 직접 구현한 도구 호출 에이전트 | RAG 성격이 강해 에이전트 역량을 보여 주는 증거로 약하다고 판단 → 멀티 에이전트·오케스트레이션·LangGraph를 배우는 별도 프로젝트 |
| 4 | [logistics-agent](https://github.com/SeonHwan/logistics-agent) | 공개 물류 데이터(KTDB 화물 OD)와 정답이 계산되는 합성 시나리오: LangGraph 멀티 에이전트(supervisor, 병렬 분석가, 취합), 검증 노드, 사람 승인, Langfuse 관측, RAGAS 채점 | 병렬로 도는 에이전트를 더 늘려 멀티 에이전트의 효과를 눈으로 확인하고, MCP 분리와 서빙도 함께 다루려고 → 새 프로젝트로 |
| 5 | [kitchen-agent](https://github.com/SeonHwan/kitchen-agent) | 가상 주방 데이터와 6명의 전문 에이전트(LangGraph), 도구의 MCP 분리, RAGAS 평가, FastAPI 서빙 | — |

### 이 프로젝트의 한계가 다음 단계로 이어진 이유
데이터와 평가 질문이 작아서(문서 4개, 질문 12개) 오케스트레이션, 답변 품질 개선, 여러 RAG 기법을 적용했을 때의 효과를 제대로 가려내기 어려웠습니다. 그래서 같은 시도를 실제 방대한 데이터로 다시 해 보려고 [dart-filing-agent](https://github.com/SeonHwan/dart-filing-agent)로 넘어갔습니다.

## 구성

| 영역 | 파일 | 하는 일 |
|---|---|---|
| 기본 파이프라인 | `chunking.py`, `embeddings.py`, `store.py`, `main.py` | 문단 단위 청킹, 임베딩, numpy 코사인 유사도 검색, 검색된 청크만 근거로 답변. 출처는 LLM이 아니라 검색 결과에서 코드가 표시 |
| 벡터DB | `qdrant_store.py` | Qdrant(임베디드/서버 모드)로 교체. payload 필터, 컬렉션별 후보 보장 |
| 검색 개선 | `hybrid_search.py`, `rerank.py`, `contextualize.py` | BM25+벡터 가중합, LLM 리랭커, 청크에 문서 내 위치를 붙이는 Contextual Retrieval |
| 평가 | `eval_dataset.py`, `metrics.py`, `run_eval.py`, `contextual_eval.py`, `ragas_eval.py` | 12개 질문, 6개 지표(precision, recall, precision@1, MRR, faithfulness, relevancy), RAGAS 비교 |
| GraphRAG | `graph_extract.py`, `graph_rag.py` | 문서에서 (개체, 관계, 개체) 트리플 추출 → 그래프 → 홉 탐색으로 다단계 질문 답변 |
| 에이전트 | `agent.py`, `langgraph_agent.py` | LLM이 검색 여부·검색어·문서 필터를 스스로 정하는 tool-use 루프(직접 구현 / LangGraph 버전) |
| MCP | `mcp_server.py`, `mcp_agent.py`, `langgraph_mcp_agent.py` | 검색을 MCP 서버로 노출하고, 에이전트가 도구를 발견해 호출 |
| 고급 RAG | `langgraph_advanced_rag.py` | Adaptive RAG 라우팅 + Self-RAG 자기반성 게이트 |
| 서비스 | `app.py`, `Dockerfile` | FastAPI(`/ask`, `/health`)로 노출 |

## 핵심 기능

### 1. 에이전트가 검색할지 말지를 스스로 판단한다
고정 RAG는 코드가 "검색 → 답변" 순서를 강제합니다. `agent.py`는 검색 도구(`search_wiki`)의 명세만 모델에게 주고,
모델이 검색 여부, 검색어, 대상 문서를 정합니다. `stop_reason`이 `end_turn`이면 종료하고 `tool_use`이면 코드가 도구를 실행해 결과를 대화에 이어 붙입니다.
같은 로직을 LangGraph의 `StateGraph`, `ToolNode`, 조건부 엣지로 다시 구성한 것이 `langgraph_agent.py`입니다.

### 2. 도구를 MCP 서버에서 발견해 호출한다
`mcp_server.py`는 검색을 MCP 서버(FastMCP, SDK v1.x)의 `search_wiki` 도구로 노출합니다. `mcp_agent.py`는 `list_tools()`로 도구 명세를 받아
모델이 부르면 `call_tool()`로 실행을 서버에 위임합니다. 에이전트는 검색이 내부에서 어떻게 동작하는지 모르고 명세만 압니다.
`langgraph_mcp_agent.py`에서는 도구 변환과 실행 위임을 `langchain-mcp-adapters`와 `ToolNode`가 대신합니다.

### 3. Adaptive RAG + Self-RAG 파이프라인
`langgraph_advanced_rag.py`는 노드 순서를 고정한 파이프라인에 품질 게이트를 붙입니다.

```
질의 분석(analyze_query)
  ├─ 잡담 → 직접 답변
  └─ 위키 질문 → wiki_pipeline(서브그래프)
        검색 → 답변 생성 → 충실도 판정(grade_faithfulness)
                              ├─ 통과 → 최종 답변
                              ├─ 근거 없는 주장이 있으면 → 재생성(최대 1회)
                              └─ 재생성 후에도 실패 → 포기(give_up)
```

질의 분석이 대상 컬렉션(`bestiary`, `patch_notes`, `quests`, `town_news`)을 0개 이상 골라 검색 범위를 좁히고(빈 리스트면 전체 검색), 검색된 후보는 LLM 리랭커가 관련도 순으로 재정렬합니다.
`wiki_pipeline`은 독립된 상태 스키마를 가진 서브그래프로 분리해, 재시도 횟수 같은 내부 필드가 부모 그래프에 노출되지 않게 했습니다.

### 4. 검색 품질을 숫자로 잰다
`run_eval.py`는 같은 12개 질문에 대해 검색기만 바꿔 끼워(`search(question, top_k)`만 있으면 됨) 6개 지표를 계산합니다.
precision@3/recall@3은 top-3을 순서 없는 집합으로 보기 때문에, 순위 개선(리랭커)을 보려고 precision@1과 MRR을 추가했습니다.
검색 지표 4개는 정답 문서 라벨과 직접 비교하고, 충실도(faithfulness)는 LLM 판정(tool calling),
답변 관련성(relevancy)은 답변에서 질문을 역으로 재구성한 뒤 원래 질문과의 임베딩 코사인 유사도로 계산합니다.
`contextual_eval.py`는 같은 평가로 Contextual Retrieval 적용 전후를 비교하고, `ragas_eval.py`는 RAGAS의 지표와 직접 구현한 지표를 비교합니다.

### 5. GraphRAG로 다단계 질문을 푼다
"탑의 시험 퀘스트를 깨면 잡는 몬스터의 재료로 방어구를 만들 수 있어?" 같은 질문은 단계마다 연결어가 달라 벡터 유사도로 각 청크를 찾기 불리합니다.
`graph_extract.py`가 문서에서 트리플을 추출하고, `graph_rag.py`가 그래프를 만들어 질문과 가까운 시작 개체를 임베딩으로 고른 뒤(기본 상위 3개) 최대 3홉까지 연결된 사실을 모아, 그 사실만 근거로 답합니다.

## 한계

- **평가셋이 작습니다.** 12개 질문, 문서 4개(합계 약 5KB)입니다. 정답 라벨은 문서 단위(`expected_source`)입니다.
- **임베딩은 로컬 모델(`BAAI/bge-m3`, dense 1024차원)로 계산합니다.** 처음 실행할 때 Hugging Face에서 모델(약 2GB)을 내려받고, CPU에서도 돌지만 느립니다. Docker 이미지에는 모델이 들어 있지 않아 첫 요청에서 내려받습니다.
- **`claude-sonnet-5`는 `temperature`를 받지 않습니다.** 그래서 같은 문서에서도 GraphRAG 트리플 추출 결과가 실행마다 달라질 수 있습니다. 추출 결과를 저장해 재사용하는 코드는 없습니다.
- Dockerfile과 CI(ruff, pytest 구조 테스트, docker build)는 있지만 실제로 배포한 적은 없습니다. CI의 pytest는 LLM/Qdrant를 호출하지 않는 그래프 구조 검증만 합니다.

## 실행

환경변수(`.env`):

| 변수 | 용도 |
|---|---|
| `ANTHROPIC_API_KEY` | 답변 생성용 키 |
| `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_BASE_URL` | (선택) Langfuse 트레이싱 |

```bash
pip install -r requirements.txt

python main.py                  # 기본 RAG 질의응답
python run_eval.py              # 6개 지표로 검색 품질 측정
python agent.py                 # tool-use 에이전트
python mcp_agent.py             # MCP 서버를 띄워 도구를 발견·호출하는 에이전트
python langgraph_advanced_rag.py   # Adaptive RAG + Self-RAG
uvicorn app:app --reload        # HTTP 서비스 (POST /ask, GET /health)
```

각 `*_demo.py`는 검색, 청킹, 임베딩, 하이브리드, 리랭커, Qdrant, MCP 클라이언트를 하나씩 눈으로 확인하는 스크립트입니다.
CI와 같은 검증은 `pip install -r requirements-dev.txt` 후 `ruff check .`, `pytest tests/`입니다.
