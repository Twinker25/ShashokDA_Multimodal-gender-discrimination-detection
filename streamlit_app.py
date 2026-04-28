import streamlit as st
import time

from src.ui import inject_css, show_results
from src.models import MODEL_CONFIGS, load_selected_model, analyze_with_sliding_window
from src.media import transcribe_audio_with_progress, transcribe_video_with_progress
from src.utils import auto_height_text_area

st.set_page_config(
    page_title="Gender Discrimination Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()


def _init_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


_init_state('selected_model_name', list(MODEL_CONFIGS.keys())[0])
_init_state('model_obj', None)
_init_state('tokenizer_obj', None)
_init_state('model_loaded', False)
_init_state('audio_transcribed_text', "")
_init_state('audio_last_filename', None)
_init_state('audio_result', None)
_init_state('video_transcribed_text', "")
_init_state('video_last_filename', None)
_init_state('video_result', None)
_init_state('text_result', None)


def load_model_for_selection(model_name):
    if st.session_state.selected_model_name != model_name or not st.session_state.model_loaded:
        prog_ph = st.empty()
        lbl_ph = st.empty()
        lbl_ph.markdown('<p class="progress-label">Loading model... 0%</p>', unsafe_allow_html=True)
        prog_ph.progress(0.0)

        model, tokenizer, success = load_selected_model(model_name)

        if success:
            st.session_state.model_obj = model
            st.session_state.tokenizer_obj = tokenizer
            st.session_state.model_loaded = True
            st.session_state.selected_model_name = model_name
            st.session_state.text_result = None
            st.session_state.audio_result = None
            st.session_state.video_result = None
            lbl_ph.markdown('<p class="progress-label">Model loaded! 100%</p>', unsafe_allow_html=True)
            prog_ph.progress(1.0)
            time.sleep(0.5)
            prog_ph.empty()
            lbl_ph.empty()
        else:
            prog_ph.empty()
            lbl_ph.empty()
            st.error(f"Failed to load **{model_name}**. Make sure model files are in the correct folder.")
            st.session_state.model_loaded = False

    return st.session_state.model_loaded


def main():
    st.markdown('<div class="main-title">Gender Discrimination Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Multimodal neural network analysis of text, audio, and video</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Settings</div>', unsafe_allow_html=True)

        model_name = st.selectbox(
            "Model",
            list(MODEL_CONFIGS.keys()),
            index=list(MODEL_CONFIGS.keys()).index(st.session_state.selected_model_name)
        )

        st.markdown("---")

        window_size = st.slider("Sliding Window Size", min_value=1, max_value=5, value=3,
                                help="Number of sentences analyzed together")
        stride = st.slider("Window Stride", min_value=1, max_value=3, value=1,
                           help="Step size for the sliding window")

        st.markdown("---")
        st.markdown("""
        <div class="info-box">
            <strong>Score &ge; 0.8:</strong> Definitely Sexist<br>
            <strong>Score 0.5–0.8:</strong> Probably Sexist<br>
            <strong>Score &lt; 0.5:</strong> Not Sexist
        </div>
        """, unsafe_allow_html=True)

    model_ready = load_model_for_selection(model_name)

    if not model_ready:
        return

    model = st.session_state.model_obj
    tokenizer = st.session_state.tokenizer_obj
    model_type = MODEL_CONFIGS[model_name]["type"]

    st.markdown('<p class="section-label">Input Method</p>', unsafe_allow_html=True)
    input_method = st.radio(
        "Input",
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
                    text_to_analyze, model, tokenizer, model_type,
                    window_size=window_size, stride=stride
                )
                st.session_state.text_result = results

        if st.session_state.text_result:
            show_results(st.session_state.text_result, model_name)
        elif not (text_to_analyze and text_to_analyze.strip()):
            st.info("Please enter text to analyze.")

    elif input_method == "Audio":
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

            if st.button("Analyze for Gender Discrimination", type="primary",
                         use_container_width=True, key="audio_analyze_btn"):
                results = analyze_with_sliding_window(
                    edited, model, tokenizer, model_type,
                    window_size=window_size, stride=stride
                )
                st.session_state.audio_result = results

        if st.session_state.audio_result:
            show_results(st.session_state.audio_result, model_name)
        elif audio_file is None and not st.session_state.audio_transcribed_text:
            st.info("Please upload an audio file to transcribe and analyze.")

    elif input_method == "Video":
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

            if st.button("Analyze for Gender Discrimination", type="primary",
                         use_container_width=True, key="video_analyze_btn"):
                results = analyze_with_sliding_window(
                    edited, model, tokenizer, model_type,
                    window_size=window_size, stride=stride
                )
                st.session_state.video_result = results

        if st.session_state.video_result:
            show_results(st.session_state.video_result, model_name)
        elif video_file is None and not st.session_state.video_transcribed_text:
            st.info("Please upload a video file to transcribe and analyze.")


if __name__ == "__main__":
    main()
