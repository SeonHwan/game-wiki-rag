from langgraph_agent import build_graph

compiled = build_graph([])
g = compiled.get_graph()

g.draw_png("graph.png")
print("graph.png 저장 완료")
