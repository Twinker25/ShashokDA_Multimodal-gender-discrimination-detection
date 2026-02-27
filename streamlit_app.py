import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import tempfile
import os
import time
import whisper
import moviepy.editor as mp

st.set_page_config(
    page_title="Gender Discrimination Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    [data-testid="stSidebar"] {
        background-color: #262730;
    }

    .sidebar-title {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #fafafa;
        padding: 0.5rem 0 1rem 0;
    }

    .stButton > button {
        background-color: #4b4b4b;
        color: #ffffff;
        border: 1px solid #6e6e6e;
        width: 100%;
    }

    .stButton > button:hover {
        background-color: #6e6e6e;
        border-color: #fafafa;
    }

    .stTextArea textarea {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #4b4b4b;
    }

    .stTextInput input {
        background-color: #262730;
        color: #fafafa;
        border: 1px solid #4b4b4b;
    }

    div[data-baseweb="select"] > div {
        background-color: #262730;
        color: #fafafa;
    }

    .stRadio label {
        color: #fafafa;
    }

    h1, h2, h3, h4 {
        color: #fafafa !important;
        text-align: center;
    }

    div.stMarkdown p {
        color: #e0e0e0;
        font-size: 1.05rem;
        text-align: center;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        color: white;
    }

    .high-risk {
        background-color: #8B0000;
        border: 2px solid #ff0000;
    }

    .medium-risk {
        background-color: #B8860B;
        border: 2px solid #ffd700;
    }

    .low-risk {
        background-color: #006400;
        border: 2px solid #00ff00;
    }

    .analysis-card {
        background: rgba(38, 39, 48, 0.8);
        border: 1px solid #4b4b4b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .metric-card {
        background: rgba(38, 39, 48, 0.8);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #4b4b4b;
        margin: 0.5rem 0;
    }

    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #e0e0e0;
        text-align: left;
    }

    .stFileUploader {
        background: rgba(38, 39, 48, 0.4);
        border-radius: 8px;
        border: 2px dashed #4b4b4b;
        padding: 1rem;
    }

    .streamlit-expanderHeader {
        background: rgba(38, 39, 48, 0.6);
        border-radius: 8px;
        border: 1px solid #4b4b4b;
    }

    .progress-label {
        text-align: center;
        color: #fafafa;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }

    [data-testid="stRadio"] > label {
        display: none;
    }

    [data-testid="stRadio"] > div {
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

MAX_LENGTH = 100
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'

def _init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default

_init_state('model', None)
_init_state('tokenizer', None)
_init_state('model_loaded', False)
_init_state('audio_transcribed_text', "")
_init_state('audio_last_filename', None)
_init_state('audio_result', None)
_init_state('video_transcribed_text', "")
_init_state('video_last_filename', None)
_init_state('video_result', None)
_init_state('text_result', None)


def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text.strip()) if s.strip()]


def prepare_text(text, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    return padded


def get_prediction(score):
    if score >= 0.8:
        return "DEFINITELY SEXIST"
    elif score >= 0.5:
        return "PROBABLY SEXIST"
    else:
        return "NOT SEXIST"


def get_confidence_level(score):
    if score >= 0.8:
        return "high"
    elif score >= 0.5:
        return "medium"
    else:
        return "low"


def get_result_class(confidence):
    if confidence == "high":
        return "high-risk"
    elif confidence == "medium":
        return "medium-risk"
    else:
        return "low-risk"


def compute_final_score(window_scores_list, sentence_scores_list):
    if not window_scores_list:
        return 0.0

    scores = [w['score'] for w in window_scores_list]
    max_score = max(scores)

    p90 = float(np.percentile(scores, 90))

    sexist_sentences = [s for s in sentence_scores_list if s['score'] >= 0.5]
    sexist_ratio = len(sexist_sentences) / max(len(sentence_scores_list), 1)

    if sexist_ratio < 0.15 and max_score < 0.85:
        isolation_penalty = 0.4 + 0.6 * sexist_ratio / 0.15
    elif sexist_ratio < 0.30 and max_score < 0.70:
        isolation_penalty = 0.7 + 0.3 * sexist_ratio / 0.30
    else:
        isolation_penalty = 1.0

    weighted_max = max_score * isolation_penalty
    final = 0.6 * p90 + 0.4 * weighted_max
    return float(np.clip(final, 0.0, 1.0))


def analyze_with_sliding_window(text, model, tokenizer, window_size=3, stride=1):
    sentences = split_sentences(text)
    if not sentences:
        return {
            'score': 0.0,
            'prediction': 'NOT SEXIST',
            'confidence': 'low',
            'window_scores': [],
            'sentences': [],
            'max_score': 0.0,
            'p90_score': 0.0,
            'sexist_ratio': 0.0
        }

    progress_placeholder = st.empty()
    label_placeholder = st.empty()

    if len(sentences) < window_size:
        label_placeholder.markdown('<p class="progress-label">Analyzing text... 0%</p>', unsafe_allow_html=True)
        progress_placeholder.progress(0)
        padded = prepare_text(text, tokenizer)
        score = float(model.predict(padded, verbose=0)[0][0])
        label_placeholder.markdown('<p class="progress-label">Analyzing text... 100%</p>', unsafe_allow_html=True)
        progress_placeholder.progress(1.0)
        time.sleep(0.3)
        progress_placeholder.empty()
        label_placeholder.empty()
        confidence = get_confidence_level(score)
        return {
            'score': score,
            'prediction': get_prediction(score),
            'confidence': confidence,
            'window_scores': [{'window': 0, 'text': text, 'score': score}],
            'sentences': [{'text': text, 'score': score}],
            'max_score': score,
            'p90_score': score,
            'sexist_ratio': 1.0 if score >= 0.5 else 0.0
        }

    total_steps = (len(sentences) - window_size) // stride + 1 + len(sentences)
    current_step = 0

    window_scores = []

    for i in range(0, len(sentences) - window_size + 1, stride):
        window = sentences[i: i + window_size]
        window_text = " ".join(window)
        padded = prepare_text(window_text, tokenizer)
        score = float(model.predict(padded, verbose=0)[0][0])
        window_scores.append({
            'window': i,
            'text': window_text,
            'score': score,
            'sentences': window
        })
        current_step += 1
        pct = current_step / total_steps
        label_placeholder.markdown(f'<p class="progress-label">Analyzing text... {pct:.0%}</p>', unsafe_allow_html=True)
        progress_placeholder.progress(pct)

    sentence_scores = []
    for sent in sentences:
        padded = prepare_text(sent, tokenizer)
        score = float(model.predict(padded, verbose=0)[0][0])
        sentence_scores.append({'text': sent, 'score': score})
        current_step += 1
        pct = current_step / total_steps
        label_placeholder.markdown(f'<p class="progress-label">Analyzing text... {pct:.0%}</p>', unsafe_allow_html=True)
        progress_placeholder.progress(min(pct, 1.0))

    label_placeholder.markdown('<p class="progress-label">Analysis complete! 100%</p>', unsafe_allow_html=True)
    progress_placeholder.progress(1.0)
    time.sleep(0.4)
    progress_placeholder.empty()
    label_placeholder.empty()

    final_score = compute_final_score(window_scores, sentence_scores)
    max_score = max(w['score'] for w in window_scores)
    p90_score = float(np.percentile([w['score'] for w in window_scores], 90))
    sexist_sentences = [s for s in sentence_scores if s['score'] >= 0.5]
    sexist_ratio = len(sexist_sentences) / max(len(sentence_scores), 1)

    confidence = get_confidence_level(final_score)
    return {
        'score': final_score,
        'prediction': get_prediction(final_score),
        'confidence': confidence,
        'window_scores': window_scores,
        'sentences': sentence_scores,
        'max_score': max_score,
        'p90_score': p90_score,
        'sexist_ratio': sexist_ratio
    }


@st.cache_resource
def load_model_and_tokenizer():
    try:
        with open('model/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        model = load_model('model/sexism_model.h5')
        return model, tokenizer, True
    except Exception:
        return None, None, False


def transcribe_audio_with_progress(audio_file):
    try:
        prog_ph = st.empty()
        lbl_ph = st.empty()

        lbl_ph.markdown('<p class="progress-label">Saving file... 10%</p>', unsafe_allow_html=True)
        prog_ph.progress(0.1)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name

        lbl_ph.markdown('<p class="progress-label">Loading Whisper model... 30%</p>', unsafe_allow_html=True)
        prog_ph.progress(0.3)
        whisper_model = whisper.load_model("base")

        lbl_ph.markdown('<p class="progress-label">Transcribing audio... 60%</p>', unsafe_allow_html=True)
        prog_ph.progress(0.6)
        result = whisper_model.transcribe(tmp_path)

        lbl_ph.markdown('<p class="progress-label">Finalizing... 90%</p>', unsafe_allow_html=True)
        prog_ph.progress(0.9)
        os.unlink(tmp_path)

        lbl_ph.markdown('<p class="progress-label">Done! 100%</p>', unsafe_allow_html=True)
        prog_ph.progress(1.0)
        time.sleep(0.4)
        prog_ph.empty()
        lbl_ph.empty()

        return result['text'], True
    except ImportError:
        return "Error: Whisper library not installed. Run: pip install openai-whisper", False
    except Exception as e:
        return f"Error converting audio: {str(e)}", False


def transcribe_video_with_progress(video_file):
    try:
        from moviepy.editor import VideoFileClip

        prog_ph = st.empty()
        lbl_ph = st.empty()

        lbl_ph.markdown('<p class="progress-label">Saving video file... 10%</p>', unsafe_allow_html=True)
        prog_ph.progress(0.1)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_file.read())
            video_path = tmp_video.name

        lbl_ph.markdown('<p class="progress-label">Extracting audio from video... 30%</p>', unsafe_allow_html=True)
        prog_ph.progress(0.3)
        video = VideoFileClip(video_path)
        audio_path = video_path.replace('.mp4', '.wav')
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()

        lbl_ph.markdown('<p class="progress-label">Loading Whisper model... 50%</p>', unsafe_allow_html=True)
        prog_ph.progress(0.5)
        whisper_model = whisper.load_model("base")

        lbl_ph.markdown('<p class="progress-label">Transcribing video... 75%</p>', unsafe_allow_html=True)
        prog_ph.progress(0.75)
        result = whisper_model.transcribe(audio_path)

        lbl_ph.markdown('<p class="progress-label">Finalizing... 90%</p>', unsafe_allow_html=True)
        prog_ph.progress(0.9)
        os.unlink(video_path)
        os.unlink(audio_path)

        lbl_ph.markdown('<p class="progress-label">Done! 100%</p>', unsafe_allow_html=True)
        prog_ph.progress(1.0)
        time.sleep(0.4)
        prog_ph.empty()
        lbl_ph.empty()

        return result['text'], True
    except ImportError:
        return "Error: Required libraries not installed. Run: pip install openai-whisper moviepy", False
    except Exception as e:
        return f"Error converting video: {str(e)}", False


def auto_height_text_area(label, value, key):
    char_per_line = 90
    wrapped_lines = 0
    for line in value.split('\n'):
        wrapped_lines += max(1, (len(line) // char_per_line) + 1)
    estimated_lines = max(wrapped_lines, 5)
    height = min(max(estimated_lines * 22 + 20, 120), 600)
    return st.text_area(label, value=value, height=height, key=key)


def show_results(results):
    st.markdown("---")
    st.markdown("## Analysis Results")

    result_class = get_result_class(results['confidence'])
    st.markdown(f"""
    <div class="result-box {result_class}">
        <div>{results['prediction']}</div>
        <div style="font-size: 1.2rem; margin-top: 0.5rem;">
            Confidence Score: {results['score']:.2%}
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Final Score", f"{results['score']:.2%}",
                  help="Weighted aggregation: 60% percentile + 40% maximum with isolation awareness")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Peak Score", f"{results.get('max_score', results['score']):.2%}",
                  help="Maximum score across all windows")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        sexist_pct = results.get('sexist_ratio', 0.0)
        st.metric("Sexist Sentences", f"{sexist_pct:.0%}",
                  help="Share of sentences with score ≥ 0.5")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Classification", results['prediction'])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Detailed Analysis")

    with st.expander("Sliding Window Analysis", expanded=True):
        st.markdown("Analysis of text segments using sliding window approach:")
        for window in results['window_scores']:
            score = window['score']
            prediction = get_prediction(score)
            color = "#ff0000" if score >= 0.8 else "#ffd700" if score >= 0.5 else "#00ff00"
            st.markdown(f"""
            <div style="background: rgba(38,39,48,0.8); padding: 1rem; border-radius: 8px;
                        margin: 0.5rem 0; border-left: 4px solid {color};">
                <strong>Window {window['window'] + 1}</strong> — Score: {score:.2%} ({prediction})<br>
                <em style="color: #d1d5db;">{window['text']}</em>
            </div>
            """, unsafe_allow_html=True)
            st.progress(score)

    with st.expander("Sentence-Level Analysis"):
        st.markdown("Individual sentence scores:")
        for idx, sent_data in enumerate(results['sentences'], 1):
            score = sent_data['score']
            prediction = get_prediction(score)
            color = "#ff0000" if score >= 0.8 else "#ffd700" if score >= 0.4 else "#00ff00"
            st.markdown(f"""
            <div style="background: rgba(38,39,48,0.8); padding: 0.75rem; border-radius: 6px;
                        margin: 0.5rem 0; border-left: 3px solid {color};">
                <strong>Sentence {idx}</strong> — {score:.2%} ({prediction})<br>
                <small style="color: #d1d5db;">{sent_data['text']}</small>
            </div>
            """, unsafe_allow_html=True)


def main():
    st.markdown("<h1 style='text-align:center;'>Gender Discrimination Detection System</h1>", unsafe_allow_html=True)

    if not st.session_state.model_loaded:
        prog_ph = st.empty()
        lbl_ph = st.empty()
        lbl_ph.markdown('<p class="progress-label">Loading neural network model... 0%</p>', unsafe_allow_html=True)
        prog_ph.progress(0.0)
        model, tokenizer, success = load_model_and_tokenizer()
        if success:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            lbl_ph.markdown('<p class="progress-label">Model loaded! 100%</p>', unsafe_allow_html=True)
            prog_ph.progress(1.0)
            time.sleep(0.5)
            prog_ph.empty()
            lbl_ph.empty()
            st.success("Model loaded successfully")
        else:
            prog_ph.empty()
            lbl_ph.empty()
            st.error("Failed to load model. Please ensure model files are in the 'model' directory.")
            return

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Settings</div>', unsafe_allow_html=True)

        window_size = st.slider(
            "Sliding Window Size",
            min_value=1, max_value=5, value=3,
            help="Number of sentences to analyze together"
        )
        stride = st.slider(
            "Window Stride",
            min_value=1, max_value=3, value=1,
            help="Step size for sliding window"
        )

        st.markdown("---")
        st.markdown("### Classification Thresholds")
        st.markdown("""
        <div class="info-box">
            <strong>Score &gt;= 0.8:</strong> Definitely Sexist<br>
            <strong>Score 0.5–0.8:</strong> Probably Sexist<br>
            <strong>Score &lt; 0.5:</strong> Not Sexist
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:center;'>Input Method</h3>", unsafe_allow_html=True)
    input_method = st.radio(
        "Choose input type:",
        ["Text", "Audio", "Video"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if input_method == "Text":
        text_to_analyze = st.text_area(
            "Enter text to analyze:",
            height=200,
            placeholder="Type or paste text here..."
        )

        if text_to_analyze and text_to_analyze.strip():
            if st.button("Analyze for Gender Discrimination", type="primary", use_container_width=True):
                results = analyze_with_sliding_window(
                    text_to_analyze,
                    st.session_state.model,
                    st.session_state.tokenizer,
                    window_size=window_size,
                    stride=stride
                )
                st.session_state.text_result = results

        if st.session_state.text_result:
            show_results(st.session_state.text_result)
        elif not (text_to_analyze and text_to_analyze.strip()):
            st.info("Please enter text to analyze.")

    elif input_method == "Audio":
        st.markdown("<h4>Upload Audio File</h4>", unsafe_allow_html=True)
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'ogg'],
            help="Upload an audio file to transcribe and analyze"
        )

        if audio_file is not None:
            current_name = audio_file.name
            if st.session_state.audio_last_filename != current_name:
                st.session_state.audio_transcribed_text = ""
                st.session_state.audio_result = None
                st.session_state.audio_last_filename = current_name

            st.audio(audio_file)
            if st.button("Transcribe Audio", type="primary", use_container_width=True):
                transcribed, success = transcribe_audio_with_progress(audio_file)
                if success:
                    st.session_state.audio_transcribed_text = transcribed
                    st.session_state.audio_result = None
                    st.success("Audio transcribed successfully!")
                else:
                    st.error(transcribed)

        if st.session_state.audio_transcribed_text:
            st.markdown("<p style='text-align:left;'><strong>Transcription:</strong></p>", unsafe_allow_html=True)
            edited = auto_height_text_area(
                "Edit transcription if needed:",
                st.session_state.audio_transcribed_text,
                key="audio_edit_area"
            )

            if st.button("Analyze for Gender Discrimination", type="primary", use_container_width=True, key="audio_analyze_btn"):
                results = analyze_with_sliding_window(
                    edited,
                    st.session_state.model,
                    st.session_state.tokenizer,
                    window_size=window_size,
                    stride=stride
                )
                st.session_state.audio_result = results

        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.audio_result:
            show_results(st.session_state.audio_result)
        elif audio_file is None and not st.session_state.audio_transcribed_text:
            st.info("Please upload an audio file to transcribe and analyze.")

    elif input_method == "Video":
        st.markdown("<h4>Upload Video File</h4>", unsafe_allow_html=True)
        video_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'mov', 'avi', 'mkv'],
            help="Upload a video file to extract audio, transcribe and analyze"
        )

        if video_file is not None:
            current_name = video_file.name
            if st.session_state.video_last_filename != current_name:
                st.session_state.video_transcribed_text = ""
                st.session_state.video_result = None
                st.session_state.video_last_filename = current_name

            st.video(video_file)
            if st.button("Transcribe Video", type="primary", use_container_width=True):
                transcribed, success = transcribe_video_with_progress(video_file)
                if success:
                    st.session_state.video_transcribed_text = transcribed
                    st.session_state.video_result = None
                    st.success("Video transcribed successfully!")
                else:
                    st.error(transcribed)

        if st.session_state.video_transcribed_text:
            st.markdown("<p style='text-align:left;'><strong>Transcription:</strong></p>", unsafe_allow_html=True)
            edited = auto_height_text_area(
                "Edit transcription if needed:",
                st.session_state.video_transcribed_text,
                key="video_edit_area"
            )

            if st.button("Analyze for Gender Discrimination", type="primary", use_container_width=True, key="video_analyze_btn"):
                results = analyze_with_sliding_window(
                    edited,
                    st.session_state.model,
                    st.session_state.tokenizer,
                    window_size=window_size,
                    stride=stride
                )
                st.session_state.video_result = results

        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.video_result:
            show_results(st.session_state.video_result)
        elif video_file is None and not st.session_state.video_transcribed_text:
            st.info("Please upload a video file to transcribe and analyze.")


if __name__ == "__main__":
    main()
