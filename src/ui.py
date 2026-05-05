import streamlit as st
from src.utils import get_prediction_label, get_result_class


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0f1117;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

.sidebar-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6366f1;
    padding-bottom: 1.2rem;
}

/* ── Buttons ── */
.stButton > button {
    background: #4f46e5;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.875rem;
    padding: 0.55rem 1.2rem;
    transition: background 0.18s, transform 0.12s, box-shadow 0.18s;
    width: 100%;
}

.stButton > button:hover {
    background: #6366f1;
    transform: translateY(-1px);
    box-shadow: 0 4px 18px rgba(99,102,241,0.35);
}

.stButton > button:active {
    transform: translateY(0);
}

/* secondary (Clear) button */
.stButton > button[kind="secondary"],
div[data-testid="column"] .stButton > button {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: #94a3b8;
    font-weight: 500;
}

.stButton > button[kind="secondary"]:hover,
div[data-testid="column"] .stButton > button:hover {
    background: rgba(255,255,255,0.1);
    color: #e2e8f0;
    transform: none;
    box-shadow: none;
}

/* ── Text inputs ── */
.stTextArea textarea,
.stTextInput input {
    background: #1e2236;
    color: #e2e8f0;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    transition: border-color 0.2s;
}

.stTextArea textarea:focus,
.stTextInput input:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.2);
    outline: none;
}

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
    background: #1e2236;
    color: #e2e8f0;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
}

/* ── Slider ── */
div[data-testid="stSlider"] .stSlider > div > div > div {
    background: #4f46e5;
}

/* ── Headings ── */
h1, h2, h3, h4 {
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
}

/* ── Page header ── */
.main-title {
    text-align: center;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #e2e8f0;
    padding-top: 1.5rem;
    padding-bottom: 0.25rem;
}

.main-title span {
    color: #818cf8;
}

.main-subtitle {
    text-align: center;
    color: #64748b;
    font-size: 0.88rem;
    margin-bottom: 2rem;
    letter-spacing: 0.01em;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.5rem;
    margin-top: 0.25rem;
}

/* ── Radio (input method tabs) ── */
[data-testid="stRadio"] > label { display: none; }
[data-testid="stRadio"] > div {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    margin-bottom: 1.5rem;
}

[data-testid="stRadio"] > div > label {
    display: flex !important;
    align-items: center;
    gap: 0.4rem;
    background: #1e2236;
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 8px;
    padding: 0.45rem 1.1rem;
    cursor: pointer;
    font-size: 0.875rem;
    font-weight: 500;
    color: #94a3b8;
    transition: all 0.18s;
}

[data-testid="stRadio"] > div > label:has(input:checked) {
    background: rgba(99,102,241,0.15);
    border-color: #6366f1;
    color: #a5b4fc;
}

/* ── File uploader ── */
.stFileUploader {
    background: rgba(255,255,255,0.02);
    border: 2px dashed rgba(99,102,241,0.25);
    border-radius: 10px;
    padding: 0.75rem;
    transition: border-color 0.2s;
}

.stFileUploader:hover {
    border-color: rgba(99,102,241,0.5);
}

/* ── Result box ── */
.result-box {
    border-radius: 12px;
    padding: 1.5rem 2rem;
    margin: 1.5rem 0;
    text-align: center;
}

.result-label {
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 0.4rem;
}

.result-score {
    font-size: 0.9rem;
    font-weight: 400;
    opacity: 0.8;
}

.high-risk {
    background: rgba(220,38,38,0.1);
    border: 1px solid rgba(220,38,38,0.35);
    color: #fca5a5;
}

.medium-risk {
    background: rgba(234,179,8,0.1);
    border: 1px solid rgba(234,179,8,0.35);
    color: #fde68a;
}

.low-risk {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.35);
    color: #86efac;
}

/* ── Metric cards ── */
.metric-card {
    background: #1e2236;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 1rem 1.1rem;
}

/* ── Analysis window items ── */
.window-item {
    background: #1a1f30;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin: 0.45rem 0;
    border-left: 3px solid transparent;
}

.window-item-text {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.3rem;
    line-height: 1.5;
}

/* ── Info / model badge ── */
.info-box {
    background: rgba(99,102,241,0.07);
    border-left: 3px solid #6366f1;
    border-radius: 6px;
    padding: 0.8rem 0.9rem;
    color: #a5b4fc;
    font-size: 0.85rem;
    line-height: 1.7;
}

.model-badge {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 0.75rem;
}

/* ── Progress label ── */
.progress-label {
    text-align: center;
    color: #818cf8;
    font-size: 0.88rem;
    margin-bottom: 4px;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 1.5rem 0;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #1e2236 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 0.875rem !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Success / error ── */
div[data-testid="stAlert"] {
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def show_results(results, model_name):
    st.markdown("---")
    st.markdown("## Analysis Results")
    st.markdown(f'<div class="model-badge">{model_name}</div>', unsafe_allow_html=True)

    result_class = get_result_class(results['confidence'])
    st.markdown(f"""
    <div class="result-box {result_class}">
        <div class="result-label">{results['prediction']}</div>
        <div class="result-score">Confidence Score: {results['score']:.2%}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Final Score", f"{results['score']:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Peak Score", f"{results.get('max_score', results['score']):.2%}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Sexist Sentences", f"{results.get('sexist_ratio', 0.0):.0%}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Classification", results['prediction'])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Detailed Analysis")

    with st.expander("Sliding Window Analysis", expanded=True):
        for window in results['window_scores']:
            score = window['score']
            prediction = get_prediction_label(score)
            color = "#ef4444" if score >= 0.8 else "#eab308" if score >= 0.5 else "#22c55e"
            st.markdown(f"""
            <div class="window-item" style="border-left-color: {color};">
                <strong>Window {window['window'] + 1}</strong> &mdash; {score:.2%} &mdash; {prediction}
                <div class="window-item-text">{window['text']}</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(score)

    with st.expander("Sentence-Level Analysis"):
        for idx, sent_data in enumerate(results['sentences'], 1):
            score = sent_data['score']
            prediction = get_prediction_label(score)
            color = "#ef4444" if score >= 0.8 else "#eab308" if score >= 0.4 else "#22c55e"
            st.markdown(f"""
            <div class="window-item" style="border-left-color: {color};">
                <strong>Sentence {idx}</strong> &mdash; {score:.2%} &mdash; {prediction}
                <div class="window-item-text">{sent_data['text']}</div>
            </div>
            """, unsafe_allow_html=True)
