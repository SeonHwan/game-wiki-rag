import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

from chunking import load_documents
from graph_extract import extract_all_triples
from graph_rag import build_graph

# 문서별로 다른 색 — 어느 문서에서 나온 관계인지 한눈에 구분되게.
SOURCE_COLORS = {
    "bestiary.md": "#4c72b0",
    "patch_notes.md": "#dd8452",
    "quests.md": "#55a868",
    "town_news.md": "#c44e52",
}
FONT = "NanumGothic"  # sudo apt install fonts-nanum 로 설치되는 폰트 이름


def main() -> None:
    documents = load_documents()
    triples = extract_all_triples(documents)
    graph: nx.MultiDiGraph = build_graph(triples)
    print(f"그래프: 노드 {graph.number_of_nodes()}개, 엣지 {graph.number_of_edges()}개")

    agraph = to_agraph(graph)
    agraph.graph_attr.update(rankdir="LR", fontname=FONT, overlap="false", splines="true")
    agraph.node_attr.update(fontname=FONT, shape="box", style="rounded,filled", fillcolor="#f0f0f0")
    agraph.edge_attr.update(fontname=FONT, fontsize="10")

    for u, v, key, data in graph.edges(keys=True, data=True):
        edge = agraph.get_edge(u, v, key=str(key))
        edge.attr["label"] = data.get("relation", "")
        edge.attr["color"] = SOURCE_COLORS.get(data.get("source", ""), "#999999")

    agraph.draw("knowledge_graph.png", prog="dot")
    print("knowledge_graph.png 저장 완료")


if __name__ == "__main__":
    main()
