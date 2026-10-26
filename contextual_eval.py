from qdrant_store import build_contextual_store, build_qdrant_store
from run_eval import METRIC_NAMES, evaluate


def main() -> None:
    plain_store = build_qdrant_store()
    contextual_store = build_contextual_store()

    variants = {
        "plain": plain_store,
        "contextual": contextual_store,
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

    plain_store.close()
    contextual_store.close()


if __name__ == "__main__":
    main()
