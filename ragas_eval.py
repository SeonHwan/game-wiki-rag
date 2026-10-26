import os
import sys
import types

_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")


class _StubChatVertexAI:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("VertexAI는 이 프로젝트에서 쓰지 않습니다")


_vertexai_stub.ChatVertexAI = _StubChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from embeddings import embed
from eval_dataset import EVAL_CASES
from main import generate_answer
from rerank import RerankStore
from store import build_store

load_dotenv()


class LocalEmbeddings(Embeddings):

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed([text])[0]


def build_ragas_dataset() -> EvaluationDataset:
    # run_eval.py의 "리랭커" 열과 apples-to-apples로 비교하려고 벡터
    # baseline이 아니라 RerankStore를 쓴다.
    store = RerankStore(build_store(), candidate_k=6)
    samples = []
    for case in EVAL_CASES:
        answer_text, results = generate_answer(store, case.question, top_k=3)
        contexts = [chunk.text for chunk, _score in results]
        samples.append(SingleTurnSample(
            user_input=case.question,
            retrieved_contexts=contexts,
            response=answer_text,
            reference=case.reference,
        ))
        print(f"  완료: {case.question}")
    return EvaluationDataset(samples=samples)


def main() -> None:
    # bypass_temperature=True: ragas는 결과를 안정적으로 재현하려고 내부적으로
    # temperature=0.01을 강제로 주입하는데, claude-sonnet-5는
    # temperature 파라미터 자체를 거부한다("deprecated for this model"). o1 계열
    # reasoning 모델을 위해 ragas가 이미 만들어둔 우회 스위치를 그대로 쓴다.
    llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-5"), bypass_temperature=True)
    embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings())

    print("검색 + 답변 생성 중 (12개 케이스, 우리 벡터 검색 재사용)...")
    dataset = build_ragas_dataset()

    print("\nragas 평가 실행 중...")
    run_config = RunConfig(max_workers=3, timeout=300)
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            ResponseRelevancy(),
        ],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    print(f"\n{'=' * 60}")
    print("ragas 결과")
    print('=' * 60)
    print(result)


if __name__ == "__main__":
    main()
