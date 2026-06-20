# app/streamlit_app.py

import streamlit as st
import time
import sys

sys.path.insert(0, ".")

from src.agent.graph import agent_graph
from src.guardrails.hallucination_checker import (
    load_ground_truth,
    check_answer,
)
from src.features.confidence_explainer import explain_confidence
from src.features.fin_sentinel import (
    analyze_ticker,
    get_drift_alerts,
)

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="QuantSense - Financial Intelligence",
    page_icon="📈",
    layout="wide",
)

# --------------------------------------------------
# Header
# --------------------------------------------------

col1, col2 = st.columns([3, 1])

with col1:
    st.title("📈 QuantSense")
    st.caption(
        "Agentic RAG for Financial Intelligence · IIT KGP Arch Hackathon 2026"
    )

with col2:
    ticker_filter = st.selectbox(
        "Focus Company",
        [
            "All",
            "AAPL",
            "JPM",
            "INFY",
            "HDFCBANK.NS",
            "RELIANCE.NS",
        ],
    )

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Pipeline Settings")

    use_hyde = st.toggle(
        "HyDE Query Expansion",
        value=True,
    )

    use_crag = st.toggle(
        "CRAG Self-Correction",
        value=True,
    )

    use_halluc = st.toggle(
        "Hallucination Guard",
        value=True,
    )

    st.divider()

    st.subheader("Management Tone Drift")

    if st.button("Analyze Management Tone"):

        if ticker_filter != "All":

            with st.spinner("Analyzing transcripts..."):

                results = analyze_ticker(ticker_filter)
                alerts = get_drift_alerts(results)

            if alerts:

                for alert in alerts:
                    st.warning(alert)

            else:
                st.success(
                    "No significant tone drift detected."
                )

# --------------------------------------------------
# Chat History
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# Ground Truth
# --------------------------------------------------

gt_df = load_ground_truth()

# --------------------------------------------------
# User Query
# --------------------------------------------------

query = st.chat_input(
    "Ask a financial question..."
)

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            start_time = time.time()

            result = agent_graph.invoke(
                {
                    "query": query,
                    "docs": [],
                    "tool_output": "",
                    "answer": "",
                    "confidence": 0.0,
                    "used_web_fallback": False,
                    "hyp_text": None,
                    "retry_count": 0,
                    "route": None,
                }
            )

            latency = round(
                time.time() - start_time,
                2,
            )

            ticker_guess = (
                ticker_filter
                if ticker_filter != "All"
                else "AAPL"
            )

            halluc = check_answer(
                result["answer"],
                ticker_guess,
                gt_df,
            )

            result["hallucination_result"] = halluc

            # --------------------------
            # Answer
            # --------------------------

            st.markdown(result["answer"])

            # --------------------------
            # Metrics
            # --------------------------

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Latency",
                f"{latency}s",
            )

            m2.metric(
                "Doc Confidence",
                f"{result.get('confidence', 0):.2f}",
            )

            m3.metric(
                "Route",
                str(
                    result.get(
                        "route",
                        "-",
                    )
                ).upper(),
            )

            m4.metric(
                "Web Fallback",
                "Yes"
                if result.get(
                    "used_web_fallback"
                )
                else "No",
            )

            # --------------------------
            # Confidence Explanation
            # --------------------------

            conf_info = explain_confidence(result)

            with st.expander(
                f"Trust Score: {conf_info['overall_pct']}%"
            ):

                for icon, message in conf_info["signals"]:

                    st.write(
                        f"{icon} {message}"
                    )

            # --------------------------
            # Sources
            # --------------------------

            if result.get("docs"):

                with st.expander(
                    f"Sources ({len(result['docs'])} documents)"
                ):

                    for idx, doc in enumerate(
                        result["docs"],
                        start=1,
                    ):

                        meta = doc["metadata"]

                        st.caption(
                            f"[{idx}] "
                            f"{meta.get('ticker')} "
                            f"{meta.get('form_type')} "
                            f"{meta.get('date')}"
                        )

                        st.info(
                            doc["text"][:400]
                        )

            # --------------------------
            # Hallucination Warnings
            # --------------------------

            if halluc.get("flagged"):

                with st.expander(
                    "Unverified Numbers"
                ):

                    for warning in halluc.get(
                        "warnings",
                        [],
                    ):

                        st.warning(warning)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
        }
    )