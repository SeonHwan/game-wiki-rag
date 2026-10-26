from eval_dataset import EVAL_CASES
from main import build_context_block, generate_answer
from metrics import (
    answer_relevancy,
    context_precision_at_k,
    context_recall,
    faithfulness,
    mrr,
    precision_at_1,
)
from qdrant_store import build_qdrant_store

METRIC_NAMES = ["precision", "recall", "precision@1", "mrr", "faithfulness", "relevancy"]


def evaluate(store, label: str) -> dict[str, float]:
    """이미 만들어진 store로 전체 평가 케이스를 돌리고, 지표별 평균을 반환."""
    print(f"\n{'#' * 60}")
    print(f"# {label}")
    print('#' * 60)

    totals = {name: 0.0 for name in METRIC_NAMES}

    for case in EVAL_CASES:
        answer_text, results = generate_answer(store, case.question, top_k=3)
        retrieved_chunks = [chunk for chunk, _score in results]
        context = build_context_block(results)

        scores = {
            "precision": context_precision_at_k(retrieved_chunks, case.expected_source),
            "recall": context_recall(retrieved_chunks, case.expected_source),
            "precision@1": precision_at_1(retrieved_chunks, case.expected_source),
            "mrr": mrr(retrieved_chunks, case.expected_source),
            "faithfulness": faithfulness(answer_text, context),
            "relevancy": answer_relevancy(case.question, answer_text),
        }
        for name, value in scores.items():
            totals[name] += value

        print(f"\nQ: {case.question}")
        print(f"  정답 출처: {case.expected_source}")
        print(f"  검색된 출처: {[c.source for c in retrieved_chunks]}")
        print("  " + "  ".join(f"{k}={v:.2f}" for k, v in scores.items()))

    n = len(EVAL_CASES)
    return {k: v / n for k, v in totals.items()}


def main() -> None:
    qdrant_store = build_qdrant_store()

    variants = {
        "Qdrant(v2)": qdrant_store,
    }

    results = {
        label: evaluate(store, label)
        for label, store in variants.items()
    }

    print(f"\n\n{'=' * 60}")
    print("최종 결과 비교 (평균)")
    print('=' * 60)
    header = "".join(f"{label:>16}" for label in variants)
    print(f"{'':<14}{header}")
    for metric in METRIC_NAMES:
        row = "".join(f"{results[label][metric]:>16.3f}" for label in variants)
        print(f"{metric:<14}{row}")

    # 임베디드 모드 종료 소음 방지: 인터프리터가 꺼지기 전에 명시적으로 닫는다.
    qdrant_store.close()


if __name__ == "__main__":
    main()
