import re
import numpy as np
import time
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LENGTH = 100
PADDING_TYPE = 'post'
TRUNC_TYPE = 'post'


def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text.strip()) if s.strip()]


def prepare_text_lstm(text, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNC_TYPE)
    return padded


def get_prediction_label(score):
    if score >= 0.8:
        return "DEFINITELY SEXIST"
    elif score >= 0.5:
        return "PROBABLY SEXIST"
    return "NOT SEXIST"


def get_confidence_level(score):
    if score >= 0.8:
        return "high"
    elif score >= 0.5:
        return "medium"
    return "low"


def get_result_class(confidence):
    if confidence == "high":
        return "high-risk"
    elif confidence == "medium":
        return "medium-risk"
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


def show_progress(label_ph, prog_ph, label, pct):
    label_ph.markdown(f'<p class="progress-label">{label}</p>', unsafe_allow_html=True)
    prog_ph.progress(pct)


def clear_progress(label_ph, prog_ph):
    time.sleep(0.4)
    prog_ph.empty()
    label_ph.empty()


def auto_height_text_area(label, value, key):
    char_per_line = 90
    wrapped_lines = 0
    for line in value.split('\n'):
        wrapped_lines += max(1, (len(line) // char_per_line) + 1)
    estimated_lines = max(wrapped_lines, 5)
    height = min(max(estimated_lines * 22 + 20, 120), 600)
    return st.text_area(label, value=value, height=height, key=key)
