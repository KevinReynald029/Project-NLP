import streamlit as st
import torch
import numpy as np
import joblib
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import zipfile
import gdown


@st.cache_resource
def download_large_models():
    if not os.path.exists("./Indobert&IndoALBERT"):
        with st.spinner("⏳ Sedang menyedot file model raksasa dari GDrive (Proses 1-3 menit)..."):
            
            id_gdrive = "1fSgGWcs4ZQbaTyyZRSf-kXD6hKnifGw3" 
            output = 'models_final.zip'
            
            gdown.download(id=id_gdrive, output=output, quiet=False)
            
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall('.')
                
            if os.path.exists(output):
                os.remove(output)
try:
    download_large_models()
except Exception as e:
    st.error(f"Gagal mengunduh model dari Google Drive Cloud. Error: {e}")


st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    page_icon="📊",
    layout="centered"
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Source+Serif+4:ital,wght@0,300;0,400;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Source Serif 4', Georgia, serif;
    background-color: #0d0d0d;
    color: #e8e0d0;
}

/* ── Main container ── */
.block-container {
    max-width: 720px !important;
    padding: 2.5rem 2rem 4rem !important;
}

/* ── Header block ── */
.header-rule {
    border: none;
    border-top: 2px solid #c9a84c;
    margin: 0 0 0.4rem 0;
}
.header-sub-rule {
    border: none;
    border-top: 1px solid #3a3225;
    margin: 0.4rem 0 1.6rem 0;
}
.app-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #e8e0d0;
    letter-spacing: 0.01em;
    line-height: 1.2;
    margin: 0.5rem 0 0.3rem 0;
}
.app-dateline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #c9a84c;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.app-subtitle {
    font-size: 0.92rem;
    font-weight: 300;
    font-style: italic;
    color: #9c9080;
    margin-bottom: 0;
}

/* ── Section labels ── */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c9a84c;
    margin-bottom: 0.4rem;
    margin-top: 1.6rem;
}

/* ── Text area ── */
textarea {
    background-color: #161410 !important;
    border: 1px solid #3a3225 !important;
    border-radius: 2px !important;
    color: #e8e0d0 !important;
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 0.95rem !important;
    caret-color: #c9a84c !important;
    padding: 0.8rem 1rem !important;
}
textarea:focus {
    border-color: #c9a84c !important;
    box-shadow: 0 0 0 1px #c9a84c22 !important;
    outline: none !important;
}
textarea::placeholder { color: #4a4238 !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background-color: #161410 !important;
    border: 1px solid #3a3225 !important;
    border-radius: 2px !important;
    color: #e8e0d0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stSelectbox > div > div:hover { border-color: #c9a84c !important; }
.stSelectbox svg { fill: #c9a84c !important; }

/* ── Button ── */
.stButton > button {
    background-color: #c9a84c !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    padding: 0.55rem 2rem !important;
    margin-top: 0.8rem;
    transition: background-color 0.15s ease, transform 0.1s ease !important;
}
.stButton > button:hover {
    background-color: #e0bd6a !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result card ── */
.result-card {
    background-color: #161410;
    border-left: 3px solid #c9a84c;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
}
.result-card-neg  { border-left-color: #c94c4c; }
.result-card-neut { border-left-color: #c9a84c; }
.result-card-pos  { border-left-color: #4caf72; }

.result-verdict {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.5rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin: 0 0 0.3rem 0;
}
.verdict-neg  { color: #e07070; }
.verdict-neut { color: #c9a84c; }
.verdict-pos  { color: #5ec47a; }

.result-confidence {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #7a7060;
    letter-spacing: 0.08em;
    margin: 0;
}
.result-model {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #4a4238;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.8rem;
    padding-top: 0.8rem;
    border-top: 1px solid #221f1a;
}

/* ── Alerts ── */
.stAlert {
    background-color: #161410 !important;
    border: 1px solid #3a3225 !important;
    border-radius: 2px !important;
    color: #9c9080 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Header ──
st.markdown('<hr class="header-rule">', unsafe_allow_html=True)
st.markdown('<p class="app-dateline">Pasar Modal · Analisis Teks · NLP</p>', unsafe_allow_html=True)
st.markdown('<h1 class="app-title">Analisis Sentimen<br>Berita Keuangan</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Klasifikasi otomatis sentimen dari judul berita pasar modal menggunakan model NLP komparatif</p>', unsafe_allow_html=True)
st.markdown('<hr class="header-sub-rule">', unsafe_allow_html=True)

# ── Load models ──
@st.cache_resource
def load_models():
    path_bert = "./Indobert&IndoALBERT/model_indobert_final/"
    tokenizer_bert = AutoTokenizer.from_pretrained(path_bert)
    model_bert = AutoModelForSequenceClassification.from_pretrained(path_bert)

    path_albert = "./Indobert&IndoALBERT/model_indoalbert_tuned/"
    tokenizer_albert = AutoTokenizer.from_pretrained(path_albert)
    model_albert = AutoModelForSequenceClassification.from_pretrained(path_albert)

    path_klasik = "./models_klasik/"
    svm_model = joblib.load(path_klasik + "svm_model.pkl")
    rf_model = joblib.load(path_klasik + "rf_model.pkl")
    tfidf_vectorizer = joblib.load(path_klasik + "tfidf_vectorizer.pkl")

    return tokenizer_bert, model_bert, tokenizer_albert, model_albert, svm_model, rf_model, tfidf_vectorizer

try:
    tokenizer_bert, model_bert, tokenizer_albert, model_albert, svm_model, rf_model, tfidf_vectorizer = load_models()
    st.markdown(
        '<p style="font-family:\'JetBrains Mono\',monospace;font-size:0.7rem;'
        'color:#4caf72;letter-spacing:0.1em;text-transform:uppercase;">'
        '● Model berhasil dimuat</p>',
        unsafe_allow_html=True
    )
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")


# ── Helper ──
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'us\$', 'dolar ', text)
    text = re.sub(r'rp\s*|rp', 'rupiah ', text)
    text = re.sub(r'%', ' persen ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Input ──
st.markdown('<p class="section-label">Teks Berita</p>', unsafe_allow_html=True)
user_input = st.text_area(
    label="",
    placeholder="Contoh: IHSG Anjlok Parah Akibat Sentimen Pasar Global...",
    height=100,
    label_visibility="collapsed"
)

st.markdown('<p class="section-label">Model Evaluasi</p>', unsafe_allow_html=True)

MODEL_OPTIONS = {
    "IndoBERT  ·  88.43%": "IndoBERT",
    "SVM Baseline  ·  78.65%": "SVM",
    "Random Forest  ·  76.00%": "RF",
    "IndoALBERT  ·  28.47%": "IndoALBERT",
}
pilihan_label = st.selectbox(
    label="",
    options=list(MODEL_OPTIONS.keys()),
    label_visibility="collapsed"
)
pilihan_model = MODEL_OPTIONS[pilihan_label]

run = st.button("Jalankan Analisis")


# ── Inference ──
if run:
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks berita terlebih dahulu.")
    else:
        text_ready = clean_text(user_input)

        label_text  = {0: "NEGATIF", 1: "NETRAL", 2: "POSITIF"}
        label_class = {0: "neg",     1: "neut",   2: "pos"}
        label_icon  = {0: "▼",       1: "◆",      2: "▲"}

        confidence_str = ""

        if pilihan_model in ("IndoBERT", "IndoALBERT"):
            tokenizer = tokenizer_bert if pilihan_model == "IndoBERT" else tokenizer_albert
            model     = model_bert     if pilihan_model == "IndoBERT" else model_albert

            inputs = tokenizer(text_ready, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=-1)
            prediction = probs.argmax().item()
            skor = probs[0][prediction].item() * 100
            confidence_str = f"Keyakinan model: {skor:.2f}%"
        else:
            text_tfidf = tfidf_vectorizer.transform([text_ready])
            pred_idx = (svm_model if pilihan_model == "SVM" else rf_model).predict(text_tfidf)[0]
            if isinstance(pred_idx, str):
                prediction = 0 if pred_idx == 'negatif' else (1 if pred_idx == 'netral' else 2)
            else:
                prediction = pred_idx

        verdict   = label_text[prediction]
        css_class = label_class[prediction]
        icon      = label_icon[prediction]

        st.markdown(f"""
        <div class="result-card result-card-{css_class}">
            <p class="result-verdict verdict-{css_class}">{icon}&nbsp; {verdict}</p>
            {"<p class='result-confidence'>" + confidence_str + "</p>" if confidence_str else ""}
            <p class="result-model">Model: {pilihan_label}</p>
        </div>
        """, unsafe_allow_html=True)