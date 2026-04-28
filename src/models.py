import pickle
import numpy as np
import streamlit as st
import time

from tensorflow.keras.models import load_model
from src.utils import (
    split_sentences, prepare_text_lstm, get_prediction_label,
    get_confidence_level, compute_final_score, show_progress, clear_progress
)

MODEL_CONFIGS = {
    "LSTM - DDET": {
        "type": "lstm",
        "model_path": "models/lstm_ddet/sexism_model.h5",
        "tokenizer_path": "models/lstm_ddet/tokenizer.pickle",
    },
    "LSTM - DSMB": {
        "type": "lstm",
        "model_path": "models/lstm_dsmb/sexism_model.h5",
        "tokenizer_path": "models/lstm_dsmb/tokenizer.pickle",
    },
    "RoBERTa - DDET": {
        "type": "roberta",
        "model_path": "models/roberta_ddet",
        "tokenizer_path": "models/roberta_ddet",
    },
    "RoBERTa - DSMB": {
        "type": "roberta",
        "model_path": "models/roberta_dsmb",
        "tokenizer_path": "models/roberta_dsmb",
    },
}


@st.cache_resource
def load_lstm(model_path, tokenizer_path):
    try:
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        model = load_model(model_path)
        return model, tokenizer, True
    except Exception:
        return None, None, False


@st.cache_resource
def load_roberta(model_path):
    try:
        from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = TFRobertaForSequenceClassification.from_pretrained(model_path)
        return model, tokenizer, True
    except Exception:
        return None, None, False


def load_selected_model(model_name):
    cfg = MODEL_CONFIGS[model_name]
    if cfg["type"] == "lstm":
        return load_lstm(cfg["model_path"], cfg["tokenizer_path"])
    return load_roberta(cfg["model_path"])


def predict_lstm(text, model, tokenizer):
    padded = prepare_text_lstm(text, tokenizer)
    return float(model.predict(padded, verbose=0)[0][0])


def predict_roberta(text, model, tokenizer, max_len=512):
    inputs = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        max_length=max_len,
        padding="max_length"
    )
    logits = model(inputs).logits
    import tensorflow as tf
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    return float(probs[1])


def predict_single(text, model, tokenizer, model_type):
    if model_type == "lstm":
        return predict_lstm(text, model, tokenizer)
    return predict_roberta(text, model, tokenizer)


def analyze_with_sliding_window(text, model, tokenizer, model_type, window_size=3, stride=1):
    sentences = split_sentences(text)

    if not sentences:
        return {
            'score': 0.0, 'prediction': 'NOT SEXIST', 'confidence': 'low',
            'window_scores': [], 'sentences': [], 'max_score': 0.0,
            'p90_score': 0.0, 'sexist_ratio': 0.0
        }

    progress_placeholder = st.empty()
    label_placeholder = st.empty()

    if len(sentences) < window_size:
        show_progress(label_placeholder, progress_placeholder, "Analyzing text... 0%", 0)
        score = predict_single(text, model, tokenizer, model_type)
        show_progress(label_placeholder, progress_placeholder, "Analyzing text... 100%", 1.0)
        clear_progress(label_placeholder, progress_placeholder)
        confidence = get_confidence_level(score)
        return {
            'score': score,
            'prediction': get_prediction_label(score),
            'confidence': confidence,
            'window_scores': [{'window': 0, 'text': text, 'score': score}],
            'sentences': [{'text': text, 'score': score}],
            'max_score': score, 'p90_score': score,
            'sexist_ratio': 1.0 if score >= 0.5 else 0.0
        }

    total_steps = (len(sentences) - window_size) // stride + 1 + len(sentences)
    current_step = 0
    window_scores = []

    for i in range(0, len(sentences) - window_size + 1, stride):
        window_text = " ".join(sentences[i: i + window_size])
        score = predict_single(window_text, model, tokenizer, model_type)
        window_scores.append({'window': i, 'text': window_text, 'score': score, 'sentences': sentences[i: i + window_size]})
        current_step += 1
        pct = current_step / total_steps
        show_progress(label_placeholder, progress_placeholder, f"Analyzing text... {pct:.0%}", pct)

    sentence_scores = []
    for sent in sentences:
        score = predict_single(sent, model, tokenizer, model_type)
        sentence_scores.append({'text': sent, 'score': score})
        current_step += 1
        pct = min(current_step / total_steps, 1.0)
        show_progress(label_placeholder, progress_placeholder, f"Analyzing text... {pct:.0%}", pct)

    show_progress(label_placeholder, progress_placeholder, "Analysis complete! 100%", 1.0)
    clear_progress(label_placeholder, progress_placeholder)

    final_score = compute_final_score(window_scores, sentence_scores)
    max_score = max(w['score'] for w in window_scores)
    p90_score = float(np.percentile([w['score'] for w in window_scores], 90))
    sexist_sentences = [s for s in sentence_scores if s['score'] >= 0.5]
    sexist_ratio = len(sexist_sentences) / max(len(sentence_scores), 1)
    confidence = get_confidence_level(final_score)

    return {
        'score': final_score,
        'prediction': get_prediction_label(final_score),
        'confidence': confidence,
        'window_scores': window_scores,
        'sentences': sentence_scores,
        'max_score': max_score,
        'p90_score': p90_score,
        'sexist_ratio': sexist_ratio
    }
