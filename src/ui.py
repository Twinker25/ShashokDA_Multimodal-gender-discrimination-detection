import streamlit as st
from src.utils import get_prediction_label, get_result_class


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0d0f14 0%, #111520 50%, #0d0f14 100%);
    color: #e8eaf0;
}

[data-testid="stSidebar"] {
    background: rgba(17, 20, 32, 0.95);
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}

.sidebar-title {
    text-align: center;
    font-size: 1.1rem;
    font-weight: 600;
    color: #a5b4fc;
    padding: 0.5rem 0 1.2rem 0;
    letter-spacing: 0.04em;
}

.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    letter-spacing: 0.03em;
    transition: all 0.2s ease;
    width: 100%;
    padding: 0.6rem 1rem;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #6366f1, #818cf8);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35);
}

.stTextArea textarea {
    background: rgba(30, 33, 52, 0.8);
    color: #e8eaf0;
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    transition: border-color 0.2s;
}

.stTextArea textarea:focus {
    border-color: rgba(99, 102, 241, 0.6);
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15);
}

div[data-baseweb="select"] > div {
    background: rgba(30, 33, 52, 0.8);
    color: #e8eaf0;
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 10px;
}

h1, h2, h3, h4 {
    color: #e8eaf0 !important;
    font-family: 'Inter', sans-serif;
}

.main-title {
    text-align: center;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a5b4fc, #818cf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
    padding-top: 1rem;
}

.main-subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 0.95rem;
    margin-bottom: 2rem;
}

.section-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #6b7280;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.result-box {
    padding: 1.5rem 2rem;
    border-radius: 14px;
    margin: 1.5rem 0;
    text-align: center;
    font-weight: 700;
    font-size: 1.4rem;
    letter-spacing: 0.03em;
}

.high-risk {
    background: linear-gradient(135deg, rgba(127,0,0,0.4), rgba(220,38,38,0.2));
    border: 1px solid rgba(220, 38, 38, 0.5);
    color: #fca5a5;
}

.medium-risk {
    background: linear-gradient(135deg, rgba(120,80,0,0.4), rgba(234,179,8,0.2));
    border: 1px solid rgba(234, 179, 8, 0.5);
    color: #fde68a;
}

.low-risk {
    background: linear-gradient(135deg, rgba(0,80,30,0.4), rgba(34,197,94,0.2));
    border: 1px solid rgba(34, 197, 94, 0.5);
    color: #86efac;
}

.metric-card {
    background: rgba(30, 33, 52, 0.6);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.3rem 0;
}

.analysis-card {
    background: rgba(30, 33, 52, 0.5);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.window-item {
    background: rgba(20, 22, 38, 0.7);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    margin: 0.5rem 0;
}

.info-box {
    background: rgba(99, 102, 241, 0.08);
    border-left: 3px solid #6366f1;
    padding: 0.85rem 1rem;
    border-radius: 6px;
    margin: 0.8rem 0;
    color: #c7d2fe;
    font-size: 0.9rem;
}

.progress-label {
    text-align: center;
    color: #a5b4fc;
    font-size: 0.9rem;
    margin-bottom: 4px;
}

.stFileUploader {
    background: rgba(30, 33, 52, 0.4);
    border-radius: 10px;
    border: 2px dashed rgba(99, 102, 241, 0.3);
    padding: 1rem;
}

[data-testid="stRadio"] > label { display: none; }
[data-testid="stRadio"] > div { justify-content: center; }

.model-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin-bottom: 1rem;
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
        <div>{results['prediction']}</div>
        <div style="font-size: 1rem; font-weight: 400; margin-top: 0.5rem; opacity: 0.85;">
            Confidence Score: {results['score']:.2%}
        </div>
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
            <div class="window-item" style="border-left: 4px solid {color};">
                <strong>Window {window['window'] + 1}</strong> &mdash; {score:.2%} ({prediction})<br>
                <span style="color: #9ca3af; font-size: 0.9rem;">{window['text']}</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(score)

    with st.expander("Sentence-Level Analysis"):
        for idx, sent_data in enumerate(results['sentences'], 1):
            score = sent_data['score']
            prediction = get_prediction_label(score)
            color = "#ef4444" if score >= 0.8 else "#eab308" if score >= 0.4 else "#22c55e"
            st.markdown(f"""
            <div class="window-item" style="border-left: 3px solid {color};">
                <strong>Sentence {idx}</strong> &mdash; {score:.2%} ({prediction})<br>
                <span style="color: #9ca3af; font-size: 0.85rem;">{sent_data['text']}</span>
            </div>
            """, unsafe_allow_html=True)
