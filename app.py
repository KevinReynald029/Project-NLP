import streamlit as st
import torch
import numpy as np
import joblib
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Financial Sentiment Analyzer", page_icon="📈", layout="centered")

st.title("📈 Aplikasi Analisis Sentimen Berita Pasar Modal")
st.write("Aplikasi ini memprediksi sentimen dari berita keuangan menggunakan 4 model komparasi.")
st.markdown("---")

@st.cache_resource
def load_models():
    path_bert = "./model_indobert_final/"
    tokenizer_bert = AutoTokenizer.from_pretrained(path_bert)
    model_bert = AutoModelForSequenceClassification.from_pretrained(path_bert)
    
    path_albert = "./model_indoalbert_tuned/"
    tokenizer_albert = AutoTokenizer.from_pretrained(path_albert)
    model_albert = AutoModelForSequenceClassification.from_pretrained(path_albert)

    path_klasik = "./models_klasik/"
    svm_model = joblib.load(path_klasik + "svm_model.pkl")
    rf_model = joblib.load(path_klasik + "rf_model.pkl")
    tfidf_vectorizer = joblib.load(path_klasik + "tfidf_vectorizer.pkl")
    
    return tokenizer_bert, model_bert, tokenizer_albert, model_albert, svm_model, rf_model, tfidf_vectorizer

try:
    tokenizer_bert, model_bert, tokenizer_albert, model_albert, svm_model, rf_model, tfidf_vectorizer = load_models()
    st.success("Success Loaded")
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'us\$', 'dolar ', text)
    text = re.sub(r'rp\s*|rp', 'rupiah ', text)
    text = re.sub(r'%', ' persen ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

user_input = st.text_area("Masukkan Judul Berita Finansial:", placeholder="Contoh: IHSG Anjlok Parah Akibat Sentimen Pasar Global...")

pilihan_model = st.selectbox(
    "Pilih Model Evaluasi:", 
    [
        "IndoBERT (88.43%)", 
        "SVM Baseline (78.65%)",
        "Random Forest (76.00%)",
        "IndoALBERT (28.47%)"
    ]
)

st.markdown("Hasil Prediksi:")

if st.button("Analisis Sentimen"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks berita terlebih dahulu.")
    else:
        text_ready = clean_text(user_input)
        labels_map = {0: "🔴 NEGATIF", 1: "🟡 NETRAL", 2: "🟢 POSITIF"}
        
        if "IndoBERT" in pilihan_model or "IndoALBERT" in pilihan_model:
            if "IndoBERT" in pilihan_model:
                tokenizer, model = tokenizer_bert, model_bert
            else:
                tokenizer, model = tokenizer_albert, model_albert
                
            inputs = tokenizer(text_ready, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits.softmax(dim=-1)
            prediction = probs.argmax().item()
            hasil_sentimen = labels_map[prediction]
            skor_keyakinan = probs[0][prediction].item() * 100
            
            st.info(f"Sentimen Terdeteksi: **{hasil_sentimen}**")
            st.caption(f"Tingkat Keyakinan Model: {skor_keyakinan:.2f}%")

        else:
            text_tfidf = tfidf_vectorizer.transform([text_ready])
            
            if "SVM" in pilihan_model:
                pred_idx = svm_model.predict(text_tfidf)[0]
            else:
                pred_idx = rf_model.predict(text_tfidf)[0]
            
            if isinstance(pred_idx, str):
                hasil_sentimen = "🔴 NEGATIF" if pred_idx == 'negatif' else "🟡 NETRAL" if pred_idx == 'netral' else "🟢 POSITIF"
            else:
                hasil_sentimen = labels_map[pred_idx]
                
            st.info(f"Sentimen Terdeteksi: **{hasil_sentimen}**")