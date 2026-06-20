# src/guardrails/citation_tracker.py
# Verifies every [Source: ...] tag in the LLM's answer against the docs that were
# actually retrieved for that query, and flags claims that carry no citation at all.
# Pairs with hallucination_checker.py — that one checks NUMBERS are right,
# this one checks SOURCES are real.

import re
from loguru import logger

CITATION_PATTERN = re.compile(r'\[Source:\s*([^\]]+)\]')
NUMERIC_PATTERN = re.compile(r'\d')


def extract_citations(answer: str) -> list:
    '''Pull every [Source: ...] tag out of the answer and split it into parts.'''
    citations = []
    for raw in CITATION_PATTERN.findall(answer):
        parts = raw.split()
        ticker = parts[0] if len(parts) > 0 else ''
        form_type = parts[1] if len(parts) > 1 else ''
        date = parts[2] if len(parts) > 2 else ''
        citations.append({'raw': raw.strip(), 'ticker': ticker, 'form_type': form_type, 'date': date})
    return citations


def build_known_sources(docs: list) -> set:
    '''(ticker, form_type) pairs that were actually retrieved for this query.'''
    known = set()
    for d in docs:
        m = d.get('metadata', {})
        known.add((m.get('ticker', ''), m.get('form_type', '')))
    return known


def validate_citations(citations: list, known_sources: set) -> dict:
    '''Split citations into valid vs fabricated based on what was actually retrieved.'''
    valid, fabricated = [], []
    for c in citations:
        key = (c['ticker'], c['form_type'])
        ticker_seen = any(c['ticker'] == k[0] for k in known_sources)
        if key in known_sources or ticker_seen:
            valid.append(c)
        else:
            fabricated.append(c)
    return {'valid': valid, 'fabricated': fabricated}


def split_sentences(text: str) -> list:
    # Good enough for report-style prose; not a full NLP sentence splitter.
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def find_uncited_claims(answer: str) -> list:
    '''Flag sentences containing a number but no [Source: ...] tag anywhere in them.'''
    uncited = []
    for sentence in split_sentences(answer):
        has_number = bool(NUMERIC_PATTERN.search(sentence))
        has_citation = bool(CITATION_PATTERN.search(sentence))
        if has_number and not has_citation:
            uncited.append(sentence)
    return uncited


def annotate_answer(answer: str, fabricated: list) -> str:
    '''Wrap fabricated citation tags so the UI can highlight them in red.'''
    annotated = answer
    for c in fabricated:
        bad_tag = f'[Source: {c["raw"]}]'
        annotated = annotated.replace(bad_tag, f'[UNVERIFIED SOURCE: {c["raw"]}]')
    return annotated


def check_citations(answer: str, docs: list) -> dict:
    '''
    Main entry point — call this right after generate_answer(), same pattern as
    hallucination_checker.check_answer().

    Returns:
        flagged: bool — True if any fabricated citation or uncited numeric claim found
        valid_citations: citations that match a doc actually retrieved this turn
        fabricated_citations: citations the LLM invented (ticker/form not in docs)
        uncited_claims: sentences with numbers but no citation at all
        coverage_pct: % of numeric-claim sentences that ARE cited
        annotated_answer: answer text with fabricated tags marked [UNVERIFIED SOURCE: ...]
    '''
    if not docs:
        logger.warning('No retrieved docs passed to citation tracker — skipping validation')
        return {
            'flagged': False, 'valid_citations': [], 'fabricated_citations': [],
            'uncited_claims': [], 'coverage_pct': 0, 'annotated_answer': answer,
        }

    citations = extract_citations(answer)
    known_sources = build_known_sources(docs)
    validated = validate_citations(citations, known_sources)
    uncited = find_uncited_claims(answer)

    numeric_sentences = [s for s in split_sentences(answer) if NUMERIC_PATTERN.search(s)]
    cited_numeric = len(numeric_sentences) - len(uncited)
    coverage_pct = round((cited_numeric / len(numeric_sentences)) * 100) if numeric_sentences else 100

    flagged = bool(validated['fabricated']) or bool(uncited)
    if flagged:
        logger.info(
            f'Citation check: {len(validated["fabricated"])} fabricated, '
            f'{len(uncited)} uncited claims, {coverage_pct}% coverage'
        )

    return {
        'flagged': flagged,
        'valid_citations': validated['valid'],
        'fabricated_citations': validated['fabricated'],
        'uncited_claims': uncited,
        'coverage_pct': coverage_pct,
        'annotated_answer': annotate_answer(answer, validated['fabricated']),
    }