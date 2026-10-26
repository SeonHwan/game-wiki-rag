import pytest
from pydantic import ValidationError

from langgraph_advanced_rag import GraphState, WikiState, build_graph


def test_build_graph_compiles_without_store():
    compiled = build_graph(None, verbose=False)
    assert compiled is not None


def test_graph_state_requires_question():
    with pytest.raises(ValidationError):
        GraphState()


def test_graph_state_defaults():
    state = GraphState(question="테스트 질문")
    assert state.route is None
    assert state.search_query == ""
    assert state.collections == []
    assert state.final_answer == ""


def test_wiki_state_requires_question():
    with pytest.raises(ValidationError):
        WikiState()


def test_wiki_state_defaults():
    state = WikiState(question="테스트 질문")
    assert state.documents == []
    assert state.generation == ""
    assert state.faithfulness_verdict is False
    assert state.unsupported_claims == []
    assert state.generation_retry_count == 0
