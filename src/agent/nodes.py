# All LangGraph node functions
import os
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv
from loguru import logger
from .tools.finance_tools import get_financial_ratios, compute_dcf, get_price_history
load_dotenv()
llm = Groq(api_key=os.getenv('GROQ_API_KEY'))
tavily = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))
TOOLS = [get_financial_ratios, compute_dcf, get_price_history]
def route_query(state: dict) -> dict:
    '''Classify query: retrieval / compute / hybrid.'''
    q = state['query']
    prompt = (f'Classify this financial query into one of: '
              f'[retrieval, compute, hybrid].\n'
              f'retrieval = needs document search only\n'
              f'compute = needs financial calculation only\n'
              f'hybrid = needs both retrieval + computation\n'
              f'Query: {q}\nAnswer with just one word.')
    resp = llm.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role':'user','content':prompt}],
        max_tokens=10
    )
    route = resp.choices[0].message.content.strip().lower()
    if route not in ['retrieval','compute','hybrid']:
        route = 'hybrid'
    logger.info(f'Routed to: {route}')
    return {**state, 'route': route}
def grade_docs(state: dict) -> dict:
    '''CRAG: Score relevance of retrieved docs. If low, trigger web fallback.'''
    docs = state.get('docs', [])
    q = state['query']
    if not docs:
        return {**state, 'confidence': 0.0}
    context = '\n'.join([d['text'][:300] for d in docs[:3]])
    prompt = (f'Rate how relevant these document excerpts are for answering:\n'
              f'QUERY: {q}\nEXCERPTS: {context}\n'
              f'Reply with a single number 0.0 to 1.0 only.')
    resp = llm.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role':'user','content':prompt}],
        max_tokens=5
    )
    try:
        conf = float(resp.choices[0].message.content.strip())
    except:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    logger.info(f'Doc confidence: {conf:.2f}')
    return {**state, 'confidence': conf}
def web_fallback(state: dict) -> dict:
    '''CRAG fallback: search web for additional context.'''
    q = state['query']
    results = tavily.search(query=q + ' financial analysis', max_results=3)
    web_docs = [{'text': r['content'], 'metadata':
        {'source': r['url'], 'ticker': 'WEB', 'form_type': 'WEB_SEARCH'}}
        for r in results.get('results', [])[:3]]
    existing = state.get('docs', [])
    return {**state, 'docs': existing + web_docs, 'used_web_fallback': True}
def run_tools(state: dict) -> dict:
    '''Run financial calculation tools based on query.'''
    q = state['query']
    # Extract ticker (simple heuristic)
    import re
    tickers = re.findall(r'\b[A-Z]{2,5}\b', q)
    tool_outputs = []
    for ticker in tickers[:2]: # Max 2 tickers per query
        if any(w in q.lower() for w in ['ratio','p/e','roe','debt']):
            tool_outputs.append(get_financial_ratios.invoke({'ticker': ticker}))
        if any(w in q.lower() for w in ['dcf','intrinsic','value']):
            tool_outputs.append(compute_dcf.invoke({'ticker': ticker}))
        if any(w in q.lower() for w in ['price','return','high','low']):
            tool_outputs.append(get_price_history.invoke({'ticker': ticker}))
    return {**state, 'tool_output': '\n'.join(tool_outputs)}
def generate_answer(state: dict) -> dict:
    '''Generate final answer with citations.'''
    docs = state.get('docs', [])
    tool_out = state.get('tool_output', '')
    context = '\n\n'.join([
        f'[Source: {d["metadata"].get("ticker","")} '
        f'{d["metadata"].get("form_type","")} '
        f'{d["metadata"].get("date","")}]\n{d["text"]}' for d in docs[:5]])
    prompt = (
        f'You are a professional financial analyst. Answer the question using '
        f'ONLY the context below. Cite sources as [Source: ...]. '
        f'If a number is from a tool calculation, say (computed).\n\n'
        f'CONTEXT:\n{context}\n\nTOOL OUTPUT:\n{tool_out}\n\n'
        f'QUESTION: {state["query"]}\n\nAnswer:'
    )
    resp = llm.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role':'user','content':prompt}],
        max_tokens=600, temperature=0.2
    )
    answer = resp.choices[0].message.content
    return {**state, 'answer': answer}