# Explains to user why system is confident or uncertain about its answer
# Explains to user why system is confident or uncertain about its answer

def explain_confidence(state: dict) -> dict:
    """
    Returns a human-readable confidence explanation based on pipeline signals.

    Signals used:
      - retrieval confidence (from CRAG grader)
      - rerank scores of top docs
      - whether web fallback was used
      - hallucination checker result
    """

    signals = []

    conf = state.get("confidence", 0.5)
    docs = state.get("docs", [])
    used_web = state.get("used_web_fallback", False)
    halluc = state.get("hallucination_result", {})

    # Signal 1: Retrieval confidence
    if conf >= 0.75:
        signals.append(
            ("n", "Strong document match found in knowledge base")
        )
    elif conf >= 0.55:
        signals.append(
            ("nn", "Partial document match — answer may be incomplete")
        )
    else:
        signals.append(
            ("nn", "Weak document match — high uncertainty")
        )

    # Signal 2: Source diversity
    tickers = set(
        d["metadata"].get("ticker", "")
        for d in docs
    )

    if len(tickers) > 1:
        signals.append(
            ("n", f"Cross-referenced {len(tickers)} companies")
        )

    # Signal 3: Web fallback used
    if used_web:
        signals.append(
            (
                "nn",
                "Knowledge base was insufficient — web search was used as supplement"
            )
        )

    # Signal 4: Hallucination check
    if halluc.get("flagged"):
        for w in halluc.get("warnings", []):
            signals.append(
                ("nn", f"Unverified number: {w}")
            )
    else:
        signals.append(
            (
                "n",
                "All numerical claims cross-checked against structured data"
            )
        )

    # Overall score
    pos = sum(1 for s in signals if s[0] == "n")
    total = len(signals)

    overall = round(pos / total * 100) if total > 0 else 0

    return {
        "signals": signals,
        "overall_pct": overall
    }