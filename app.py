from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langgraph_advanced_rag import answer, build_graph
from qdrant_store import build_multi_collection_store
from rerank import RerankStore

load_dotenv()

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # RerankStore로 감싸야 함 — MultiCollectionStore.search()는 이제 collection마다
    # top_k개씩(자르지 않고) 반환하므로, 감싸지 않으면 top_k보다 훨씬 많은 문서가
    # 그대로 generate_answer에 들어간다(qdrant_store.py의 MultiCollectionStore.search
    # docstring 참고). RerankStore가 LLM으로 재정렬 후 최종 top_k만 추려준다.
    store = RerankStore(build_multi_collection_store(), candidate_k=10)
    _state["store"] = store
    _state["graph"] = build_graph(store, verbose=False)
    yield
    store.close()


app = FastAPI(lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    return AskResponse(answer=answer(_state["graph"], request.question))
