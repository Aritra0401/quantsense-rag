# LangGraph agent — wires all nodes together
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional
from .nodes import route_query, grade_docs, web_fallback, run_tools, generate_answer
from ..retrieval.hybrid_merger import load_indexes, hybrid_search
from ..retrieval.reranker import rerank
class AgentState(TypedDict):
    query: str
    route: Optional[str]
    docs: List[dict]
    tool_output: str
    answer: str
    confidence: float
    used_web_fallback: bool
    hyp_text: Optional[str]
    retry_count: int
# Load indexes once at startup
bm25_data, col, embed_model = load_indexes()
def retrieve_node(state: AgentState) -> AgentState:
    results, hyp = hybrid_search(
        state['query'],
        bm25_data,
        col,
        embed_model
    )

    print("\n" + "=" * 60)
    print("TOP RETRIEVED DOCUMENTS")
    print("=" * 60)

    for i, r in enumerate(results[:5], 1):
        print(f"\nResult {i}")

        metadata = r.get("metadata", {})

        print("Ticker    :", metadata.get("ticker"))
        print("Form Type :", metadata.get("form_type"))
        print("Date      :", metadata.get("date"))
        print("Source    :", metadata.get("source_file"))

        preview = r.get("text", "")[:200]
        print("Preview   :", preview)

    print("=" * 60 + "\n")

    # Reranker temporarily disabled
    reranked = results[:5]

    return {
        **state,
        'docs': reranked,
        'hyp_text': hyp
    }
def decide_after_route(state: AgentState) -> str:
    route = state.get('route', 'hybrid')
    if route == 'compute': return 'tools'
    return 'retriever'
def decide_after_grade(state: AgentState) -> str:
    conf = state.get('confidence', 0.5)
    retries = state.get('retry_count', 0)
    if conf < 0.55 and retries < 1:
        return 'web_fallback' # CRAG loop
    return 'generate'
def build_graph():
    g = StateGraph(AgentState)
    g.add_node('router', route_query)
    g.add_node('retriever', retrieve_node)
    g.add_node('grader', grade_docs)
    g.add_node('web_fallback', web_fallback)
    g.add_node('tools', run_tools)
    g.add_node('generate', generate_answer)
    g.set_entry_point('router')
    g.add_conditional_edges('router', decide_after_route,
        {'retriever':'retriever','tools':'tools'})
    g.add_edge('retriever', 'grader')
    g.add_conditional_edges('grader', decide_after_grade,
        {'web_fallback':'web_fallback','generate':'generate'})
    g.add_edge('web_fallback', 'generate')
    g.add_edge('tools', 'generate')
    g.add_edge('generate', END)
    return g.compile()
# Singleton graph
agent_graph = build_graph()
