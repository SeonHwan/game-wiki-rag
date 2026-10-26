import pygraphviz as pgv

from langgraph_advanced_rag import build_graph

compiled = build_graph(None, verbose=False)
g = compiled.get_graph()

viz = pgv.AGraph(directed=True, nodesep=0.9, ranksep=1.0, bgcolor="black")

for node in g.nodes:
    viz.add_node(
        node,
        label=f"<<B>{node}</B>>",
        style="filled",
        fillcolor="yellow",
        fontsize=15,
        fontname="arial",
    )

for start, end, data, cond in g.edges:
    label = str(data) if data is not None else None
    viz.add_edge(
        start,
        end,
        label=f"<<U>{label}</U>>" if label else "",
        fontsize=12,
        fontname="arial",
        fontcolor="white",
        color="white",
        style="dotted" if cond else "solid",
    )

if first := g.first_node():
    viz.get_node(first.id).attr.update(fillcolor="lightblue")
if last := g.last_node():
    viz.get_node(last.id).attr.update(fillcolor="orange")

viz.draw("graph.png", format="png", prog="dot")
viz.close()
print("graph.png 저장 완료 (배경 검정)")
