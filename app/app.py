import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import streamlit as st
import plotly.express as px

from core.data_loader import load_all, load_order_book, save_order_book
from core.llm_orchestrator import handle_user_request
from core.report_export import export_to_pdf

st.set_page_config(page_title="Production Scheduler", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="st-"], [class*="css"] {
        font-family: 'Times New Roman', Times, serif !important;
    }
    html, body, p, span, label, div, h5, h6 {
        color: #F1F5F9 !important;
    }
    .stApp {
        background-color: #0B0F19;
        overflow-x: hidden;
    }
    .grid-bg {
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        overflow: hidden;
        background-color: #0B0F19;
    }
    .grid-bg::before {
        content: "";
        position: absolute;
        left: -50%;
        right: -50%;
        top: 35%;
        bottom: -50%;
        background-image:
            linear-gradient(rgba(99, 102, 241, 0.35) 1px, transparent 1px),
            linear-gradient(90deg, rgba(99, 102, 241, 0.35) 1px, transparent 1px);
        background-size: 48px 48px;
        transform: perspective(400px) rotateX(62deg);
        transform-origin: top;
    }
    .grid-bg::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(to bottom, #0B0F19 0%, #0B0F19 32%, transparent 55%, transparent 100%);
    }
    .block-container {
        position: relative;
        z-index: 1;
        background-color: rgba(30, 41, 59, 0.88);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 18px;
        padding: 2rem 2.5rem 3rem !important;
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        margin-top: 1.5rem;
    }
    h1 {
        color: #F1F5F9 !important;
        text-align: center !important;
    }
    h2, h3, h4 {
        color: #6366F1 !important;
        text-align: left !important;
    }
    .stButton button, .stDownloadButton button {
        background-color: #10B981 !important;
        color: #F1F5F9 !important;
        border: none !important;
        font-weight: bold;
    }
    #MainMenu, footer, [data-testid="stToolbar"] {
        visibility: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="grid-bg"></div>', unsafe_allow_html=True)

data = load_all()

if "orders" not in st.session_state:
    st.session_state.orders = load_order_book()

title_col, book_col = st.columns([5, 1])
with title_col:
    st.title("Multi-Agent Production Scheduler")
with book_col:
    show_order_book = st.checkbox("Order Book")

if show_order_book:
    st.dataframe(st.session_state.orders, width="stretch")

st.subheader("New Order Request")
prompt = st.text_input("Describe the order", placeholder="I need 150 units of Bracket by Friday")
priority_choice = st.segmented_control(
    "Priority (required -- you must choose, there is no default)",
    options=["High", "Medium", "Low"],
    default=None,
)
if st.button("Submit Order"):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("No ANTHROPIC_API_KEY found. Add it to a .env file in the project root and restart the app.")
    elif not prompt.strip():
        st.warning("Type a request first.")
    elif priority_choice is None:
        st.warning("Priority is required -- please select High, Medium, or Low before submitting.")
    else:
        try:
            with st.spinner("Parsing request and re-planning..."):
                result = handle_user_request(prompt, priority_choice, st.session_state.orders, data["routing"], data["machines"], data["energy"])
        except Exception:
            st.error(
                "Something went wrong while processing this request. This can happen if the request is "
                "ambiguous or doesn't map to a known product -- try rephrasing it (e.g. include a clear "
                "product name, quantity, and timeframe)."
            )
            result = None

        if result is not None:
            if result.get("error"):
                st.error(result["report"])
            else:
                st.session_state.orders = result["updated_orders"]
                save_order_book(st.session_state.orders)
                st.session_state.last_result = result

if "last_result" in st.session_state:
    result = st.session_state.last_result
    st.success(f"Order **{result['new_order_id']}** added to the order book.")
    st.markdown(result["summary"].replace("$", "\\$"))
    st.markdown(f"> {result['comparison_note']}")

    st.subheader("Generated Schedule (Gantt view)")
    fig = px.timeline(
        result["schedule"],
        x_start="start",
        x_end="end",
        y="machine_id",
        color="order_id",
        hover_data=["product_id", "capability", "priority", "cost", "co2_kg"],
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#eef6fb",
    )
    st.plotly_chart(fig, width="stretch")

    pdf_report_text = result["report"] + "\n\n" + result["comparison_note"]
    pdf_bytes = export_to_pdf(
        result["schedule"], result["metrics"], pdf_report_text,
        title=f"Production Plan Report -- Order {result['new_order_id']}",
    )
    st.download_button(
        "Download PDF report", pdf_bytes, file_name=f"{result['new_order_id']}_report.pdf",
        mime="application/pdf",
    )
