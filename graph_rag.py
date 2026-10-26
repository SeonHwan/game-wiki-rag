from collections import defaultdict

import anthropic
import networkx as nx
import numpy as np
from pydantic import BaseModel

from embeddings import embed
from graph_extract import Triple

MODEL = "claude-sonnet-5"


def _normalize(name: str) -> str:
    return name.split("(")[0].strip()


class _EntityGroup(BaseModel):
    canonical: str  # 대표로 쓸 이름
    aliases: list[str]  # canonical과 같은 개체를 가리키는 "다른" 이름들(canonical 제외)


class _ResolutionResult(BaseModel):
    groups: list[_EntityGroup]


RESOLVE_TOOL = {
    "name": "report_entity_groups",
    "description": "같은 실제 개체를 가리키는 서로 다른 이름들을 그룹으로 묶어 반환한다.",
    "input_schema": _ResolutionResult.model_json_schema(),
}


def resolve_entities(triples: list[Triple], max_examples_per_node: int = 3) -> list[Triple]:
    normalized = [
        Triple(subject=_normalize(t.subject), relation=t.relation, object=_normalize(t.object), source=t.source)
        for t in triples
    ]

    summaries: dict[str, list[str]] = defaultdict(list)
    for t in normalized:
        line = f"({t.subject} -{t.relation}-> {t.object})"
        summaries[t.subject].append(line)
        summaries[t.object].append(line)

    node_list = "\n".join(
        f"- {name}: {'; '.join(examples[:max_examples_per_node])}"
        for name, examples in sorted(summaries.items())
    )

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        # temperature는 주지 않는다(claude-sonnet-5가 받지 않음) — extract_triples()와 같다.
        system=(
            "아래 [개체 목록]은 여러 문서에서 독립적으로 추출된 개체 이름과, 그 개체가 "
            "등장하는 관계 몇 개(맥락 파악용)입니다. 서로 다른 이름이지만 같은 실제 "
            "개체를 가리키는 것들을 찾아 그룹으로 묶으세요 — 이름만 봐서 애매하면 "
            "옆의 관계 맥락을 참고하세요. canonical은 더 명확하고 일반적인 이름으로 "
            "고르세요. 확실하지 않으면 묶지 마세요 — 서로 다른 개체를 잘못 합치는 "
            "쪽이 안 합치는 쪽보다 더 나쁩니다."
        ),
        messages=[{"role": "user", "content": f"[개체 목록]\n{node_list}"}],
        tools=[RESOLVE_TOOL],
        tool_choice={"type": "tool", "name": "report_entity_groups"},
    )
    tool_block = next(b for b in message.content if b.type == "tool_use")
    result = _ResolutionResult.model_validate(tool_block.input)

    rename: dict[str, str] = {}
    for group in result.groups:
        for alias in group.aliases:
            if alias != group.canonical:
                rename[alias] = group.canonical
    if rename:
        print(f"  🔗 개체 병합: {rename}")

    return [
        Triple(
            subject=rename.get(t.subject, t.subject),
            relation=t.relation,
            object=rename.get(t.object, t.object),
            source=t.source,
        )
        for t in normalized
    ]


def build_graph(triples: list[Triple]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for t in triples:
        graph.add_edge(_normalize(t.subject), _normalize(t.object), relation=t.relation, source=t.source)
    return graph


class _SeedResult(BaseModel):
    entities: list[str]


SEED_TOOL = {
    "name": "report_seeds",
    "description": "질문과 관련된 탐색 시작 개체(노드) 목록을 반환한다.",
    "input_schema": _SeedResult.model_json_schema(),
}


def embed_nodes(graph: nx.MultiDiGraph) -> dict[str, list[float]]:
    nodes = list(graph.nodes)
    vectors = embed(nodes)
    return dict(zip(nodes, vectors))


def find_seed_entities_embedding(
    question: str,
    node_embeddings: dict[str, list[float]],
    top_k: int = 3,
) -> list[str]:
    query_vec = np.array(embed([question])[0])
    scored = []
    for name, vec in node_embeddings.items():
        v = np.array(vec)
        score = float(query_vec @ v / (np.linalg.norm(query_vec) * np.linalg.norm(v)))
        scored.append((name, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _score in scored[:top_k]]


def find_seed_entities_llm(question: str, graph: nx.MultiDiGraph) -> list[str]:
    node_list = "\n".join(sorted(graph.nodes))

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=(
            "아래 [개체 목록]에 있는 이름 중, [질문]에 답하기 위해 그래프 탐색을 "
            "시작해야 할 개체를 고르세요. 목록에 있는 이름을 정확히 그대로 쓰세요. "
            "질문에 그 개체 이름이 직접 나오지 않아도, 질문 내용상 관련될 개체를 "
            "추론해서 고르세요(예: '그 퀘스트를 깨면 잡는 몬스터'처럼 우회적으로 "
            "언급된 경우, 그 퀘스트 이름 자체를 시작점으로 고르면 됨)."
        ),
        messages=[{
            "role": "user",
            "content": f"[개체 목록]\n{node_list}\n\n[질문]\n{question}",
        }],
        tools=[SEED_TOOL],
        tool_choice={"type": "tool", "name": "report_seeds"},
    )
    tool_block = next(b for b in message.content if b.type == "tool_use")
    result = _SeedResult.model_validate(tool_block.input)
    # 방어적 필터 — LLM이 목록에 없는 이름을 낼 수도 있음(그래프에 실제로 있는 것만 채택)
    return [e for e in result.entities if e in graph.nodes]


def traverse(graph: nx.MultiDiGraph, seeds: list[str], max_hops: int = 3) -> list[Triple]:
    visited: set[str] = set(seeds)
    frontier: set[str] = set(seeds)
    collected: list[Triple] = []
    seen_edges: set[tuple] = set()

    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for node in frontier:
            if node not in graph:
                continue
            for _, target, data in graph.out_edges(node, data=True):
                key = (node, data["relation"], target, data["source"])
                if key not in seen_edges:
                    seen_edges.add(key)
                    collected.append(Triple(subject=node, relation=data["relation"], object=target, source=data["source"]))
                if target not in visited:
                    next_frontier.add(target)
            for source_node, _, data in graph.in_edges(node, data=True):
                key = (source_node, data["relation"], node, data["source"])
                if key not in seen_edges:
                    seen_edges.add(key)
                    collected.append(Triple(subject=source_node, relation=data["relation"], object=node, source=data["source"]))
                if source_node not in visited:
                    next_frontier.add(source_node)
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    return collected


def format_facts(facts: list[Triple]) -> str:
    return "\n".join(f"- {t.subject} → {t.relation} → {t.object} (출처: {t.source})" for t in facts)


GENERATE_SYSTEM = (
    "당신은 게임 위키 Q&A 어시스턴트입니다. 주어진 [관계 사실]만 근거로 한국어로 "
    "간결하게(2~4문장) 답하세요. 사실에 없는 내용은 지어내지 말고, 부족하면 "
    "'주어진 사실로는 알 수 없습니다'라고 하세요."
)


def graph_answer(
    question: str,
    graph: nx.MultiDiGraph,
    node_embeddings: dict[str, list[float]],
    max_hops: int = 3,
    seed_top_k: int = 3,
    verbose: bool = True,
) -> str:
    seeds = find_seed_entities_embedding(question, node_embeddings, top_k=seed_top_k)
    if verbose:
        print(f"  🌱 seed 개체: {seeds}")
    facts = traverse(graph, seeds, max_hops=max_hops)
    if verbose:
        print(f"  🕸️  탐색된 관계 사실 {len(facts)}개:")
        for f in facts:
            print(f"     - {f.subject} → {f.relation} → {f.object} ({f.source})")

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=GENERATE_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"[관계 사실]\n{format_facts(facts)}\n\n[질문]\n{question}",
        }],
    )
    return next(b.text for b in message.content if b.type == "text").strip()
