import os
import tempfile
import time
import streamlit as st
import whisper

from src.utils import show_progress, clear_progress


def transcribe_audio_with_progress(audio_file):
    try:
        prog_ph = st.empty()
        lbl_ph = st.empty()

        show_progress(lbl_ph, prog_ph, "Saving file... 10%", 0.1)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name

        show_progress(lbl_ph, prog_ph, "Loading Whisper model... 30%", 0.3)
        whisper_model = whisper.load_model("base")

        show_progress(lbl_ph, prog_ph, "Transcribing audio... 60%", 0.6)
        result = whisper_model.transcribe(tmp_path)

        show_progress(lbl_ph, prog_ph, "Finalizing... 90%", 0.9)
        os.unlink(tmp_path)

        show_progress(lbl_ph, prog_ph, "Done! 100%", 1.0)
        clear_progress(lbl_ph, prog_ph)

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

        show_progress(lbl_ph, prog_ph, "Saving video file... 10%", 0.1)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_file.read())
            video_path = tmp_video.name

        show_progress(lbl_ph, prog_ph, "Extracting audio from video... 30%", 0.3)
        video = VideoFileClip(video_path)
        audio_path = video_path.replace('.mp4', '.wav')
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()

        show_progress(lbl_ph, prog_ph, "Loading Whisper model... 50%", 0.5)
        whisper_model = whisper.load_model("base")

        show_progress(lbl_ph, prog_ph, "Transcribing video... 75%", 0.75)
        result = whisper_model.transcribe(audio_path)

        show_progress(lbl_ph, prog_ph, "Finalizing... 90%", 0.9)
        os.unlink(video_path)
        os.unlink(audio_path)

        show_progress(lbl_ph, prog_ph, "Done! 100%", 1.0)
        clear_progress(lbl_ph, prog_ph)

        return result['text'], True
    except ImportError:
        return "Error: Required libraries not installed. Run: pip install openai-whisper moviepy", False
    except Exception as e:
        return f"Error converting video: {str(e)}", False
