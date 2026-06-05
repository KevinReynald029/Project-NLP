import streamlit as st
import torch
import numpy as np
import joblib
import re
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DistilBertTokenizer
)
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📊",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Serif+4:ital,wght@0,300;0,400;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
    background-color: #0d0d0d;
    color: #e8e0d0;
}
.block-container { max-width: 720px !important; padding: 2.5rem 2rem 4rem !important; }
.header-rule { border: none; border-top: 2px solid #c9a84c; margin: 0 0 0.4rem 0; }
.header-sub-rule { border: none; border-top: 1px solid #3a3225; margin: 0.4rem 0 1.6rem 0; }
.app-title { font-family: 'Playfair Display', Georgia, serif; font-size: 2.1rem; font-weight: 700; color: #e8e0d0; letter-spacing: 0.01em; line-height: 1.2; margin: 0.5rem 0 0.3rem 0; }
.app-dateline { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #c9a84c; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.2rem; }
.app-subtitle { font-size: 0.92rem; font-weight: 300; font-style: italic; color: #9c9080; margin-bottom: 0; }
.section-label { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase; color: #c9a84c; margin-bottom: 0.4rem; margin-top: 1.6rem; }
textarea { background-color: #161410 !important; border: 1px solid #3a3225 !important; border-radius: 2px !important; color: #e8e0d0 !important; font-family: 'Source Serif 4', Georgia, serif !important; font-size: 0.95rem !important; caret-color: #c9a84c !important; padding: 0.8rem 1rem !important; }
textarea:focus { border-color: #c9a84c !important; box-shadow: 0 0 0 1px #c9a84c22 !important; outline: none !important; }
textarea::placeholder { color: #4a4238 !important; }
.stSelectbox > div > div { background-color: #161410 !important; border: 1px solid #3a3225 !important; border-radius: 2px !important; color: #e8e0d0 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.82rem !important; }
.stSelectbox > div > div:hover { border-color: #c9a84c !important; }
.stSelectbox svg { fill: #c9a84c !important; }
.stButton > button { background-color: #c9a84c !important; color: #0d0d0d !important; border: none !important; border-radius: 2px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.75rem !important; letter-spacing: 0.14em !important; text-transform: uppercase !important; font-weight: 500 !important; padding: 0.55rem 2rem !important; margin-top: 0.8rem; transition: background-color 0.15s ease, transform 0.1s ease !important; }
.stButton > button:hover { background-color: #e0bd6a !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }
.result-card { background-color: #161410; border-left: 3px solid #c9a84c; padding: 1.2rem 1.4rem; margin-top: 1rem; }
.result-card-neg  { border-left-color: #c94c4c; }
.result-card-neut { border-left-color: #c9a84c; }
.result-card-pos  { border-left-color: #4caf72; }
.result-verdict { font-family: 'Playfair Display', Georgia, serif; font-size: 1.5rem; font-weight: 600; letter-spacing: 0.02em; margin: 0 0 0.3rem 0; }
.verdict-neg  { color: #e07070; }
.verdict-neut { color: #c9a84c; }
.verdict-pos  { color: #5ec47a; }
.result-confidence { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #7a7060; letter-spacing: 0.08em; margin: 0; }
.result-model { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #4a4238; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px solid #221f1a; }
.stAlert { background-color: #161410 !important; border: 1px solid #3a3225 !important; border-radius: 2px !important; color: #9c9080 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

st.markdown('<hr class="header-rule">', unsafe_allow_html=True)
st.markdown('<p class="app-dateline">Pasar Modal · Analisis Teks · NLP</p>', unsafe_allow_html=True)
st.markdown('<h1 class="app-title">Analisis Sentimen<br>Berita Keuangan</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Klasifikasi otomatis sentimen dari judul berita pasar modal menggunakan model NLP komparatif</p>', unsafe_allow_html=True)
st.markdown('<hr class="header-sub-rule">', unsafe_allow_html=True)

HF_INDOBERT       = "KevinRey/nlp-indobert-sentiment"
HF_INDODISTILBERT = "KevinRey/nlp-indodistilbert-sentiment"

@st.cache_resource
def load_models():
    tokenizer_bert = AutoTokenizer.from_pretrained(HF_INDOBERT)
    model_bert     = AutoModelForSequenceClassification.from_pretrained(HF_INDOBERT)
    model_bert.eval()

    tokenizer_distil = DistilBertTokenizer.from_pretrained(HF_INDODISTILBERT)
    model_distil     = AutoModelForSequenceClassification.from_pretrained(HF_INDODISTILBERT)
    model_distil.eval()

    svm_model = joblib.load(
        hf_hub_download(f"KevinRey/nlp-classical-sentiment", "svm_model.pkl")
    )
    rf_model = joblib.load(
        hf_hub_download(f"KevinRey/nlp-classical-sentiment", "rf_model.pkl")
    )
    tfidf_vectorizer = joblib.load(
        hf_hub_download(f"KevinRey/nlp-classical-sentiment", "tfidf_vectorizer.pkl")
    )

    return (
        tokenizer_bert, model_bert,
        tokenizer_distil, model_distil,
        svm_model, rf_model, tfidf_vectorizer
    )

try:
    (
        tokenizer_bert, model_bert,
        tokenizer_distil, model_distil,
        svm_model, rf_model, tfidf_vectorizer
    ) = load_models()

    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;'
        'color:#4caf72;letter-spacing:0.1em;text-transform:uppercase;">'
        '● Semua model berhasil dimuat</p>',
        unsafe_allow_html=True
    )
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")
    st.stop()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'us\$', 'dolar ', text)
    text = re.sub(r'rp\s*', 'rupiah ', text)
    text = re.sub(r'%', ' persen ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_transformer(text, tokenizer, model, is_distilbert=False):
    """Inference untuk IndoBERT dan IndoDistilBERT."""
    kwargs = dict(truncation=True, max_length=64, return_tensors="pt")
    if is_distilbert:
        kwargs["return_token_type_ids"] = False
    inputs = tokenizer(text, **kwargs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
    probs      = logits.softmax(dim=-1)
    prediction = probs.argmax().item()
    confidence = probs[0][prediction].item() * 100
    return prediction, confidence

def predict_classical(text, model, vectorizer):
    """Inference untuk SVM dan Random Forest."""
    X = vectorizer.transform([text])
    pred_raw = model.predict(X)[0]
    if isinstance(pred_raw, str):
        prediction = {'negatif': 0, 'netral': 1, 'positif': 2}.get(pred_raw, 1)
    else:
        prediction = int(pred_raw)
    try:
        probs      = model.predict_proba(X)[0]
        confidence = probs.max() * 100
    except AttributeError:
        confidence = None
    return prediction, confidence

st.markdown('<p class="section-label">Teks Berita</p>', unsafe_allow_html=True)
user_input = st.text_area(
    label="",
    placeholder="Contoh: IHSG Anjlok Parah Akibat Sentimen Pasar Global...",
    height=100,
    label_visibility="collapsed"
)

st.markdown('<p class="section-label">Model Evaluasi</p>', unsafe_allow_html=True)

MODEL_OPTIONS = {
    "IndoDistilBERT  ·  Macro-F1 88.18%": "IndoDistilBERT",
    "IndoBERT        ·  Macro-F1 87.81%": "IndoBERT",
    "SVM             ·  Macro-F1 83.20%": "SVM",
    "Random Forest   ·  Macro-F1 77.91%": "RF",
}
pilihan_label = st.selectbox(
    label="",
    options=list(MODEL_OPTIONS.keys()),
    label_visibility="collapsed"
)
pilihan_model = MODEL_OPTIONS[pilihan_label]

run = st.button("Jalankan Analisis")

if run:
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks berita terlebih dahulu.")
    else:
        text_ready = clean_text(user_input)

        label_text  = {0: "NEGATIF", 1: "NETRAL",  2: "POSITIF"}
        label_class = {0: "neg",     1: "neut",     2: "pos"}
        label_icon  = {0: "▼",       1: "◆",        2: "▲"}

        with st.spinner("Menganalisis..."):
            if pilihan_model == "IndoBERT":
                prediction, confidence = predict_transformer(
                    text_ready, tokenizer_bert, model_bert, is_distilbert=False
                )
                confidence_str = f"Keyakinan model: {confidence:.2f}%"

            elif pilihan_model == "IndoDistilBERT":
                prediction, confidence = predict_transformer(
                    text_ready, tokenizer_distil, model_distil, is_distilbert=True
                )
                confidence_str = f"Keyakinan model: {confidence:.2f}%"

            else:
                model_dipilih = svm_model if pilihan_model == "SVM" else rf_model
                prediction, confidence = predict_classical(
                    text_ready, model_dipilih, tfidf_vectorizer
                )
                confidence_str = (
                    f"Keyakinan model: {confidence:.2f}%"
                    if confidence is not None
                    else "Keyakinan model: N/A (Prediksi Mutlak)"
                )

        verdict   = label_text[prediction]
        css_class = label_class[prediction]
        icon      = label_icon[prediction]

        st.markdown(f"""
        <div class="result-card result-card-{css_class}">
            <p class="result-verdict verdict-{css_class}">{icon}&nbsp; {verdict}</p>
            <p class="result-confidence">{confidence_str}</p>
            <p class="result-model">Model: {pilihan_label}</p>
        </div>
        """, unsafe_allow_html=True)