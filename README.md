# Financial Sentiment Analyzer

A Natural Language Processing (NLP) application for classifying the sentiment of **Indonesian financial news headlines** into **Negative, Neutral, or Positive** sentiment.

The project compares transformer-based and classical machine learning approaches and provides an interactive **Streamlit** interface so users can test each model directly.

**Live Demo:** https://analysis-sentimen-ekonomi.streamlit.app

---

## Features

- Classifies Indonesian financial news into **Negative, Neutral, and Positive** sentiment
- Supports four different models for comparison:
  - **IndoDistilBERT** — Macro-F1 88.18%
  - **IndoBERT** — Macro-F1 87.81%
  - **SVM** — Macro-F1 83.20%
  - **Random Forest** — Macro-F1 77.91%
- Displays prediction confidence when supported by the selected model
- Includes text preprocessing for common financial expressions such as `US$`, `Rp`, and `%`
- Downloads trained models from Hugging Face Hub when the application starts
- Provides a simple web interface built with Streamlit

---

## How It Works

1. The user enters an Indonesian financial news headline.
2. The text is cleaned and normalized before inference.
3. The user selects one of the available NLP models.
4. The selected model predicts one of three sentiment classes: **Negative**, **Neutral**, or **Positive**.
5. The application displays the predicted sentiment and confidence score when available.

---

## Models

| Model | Approach | Macro-F1 |
|---|---|---:|
| IndoDistilBERT | Transformer | **88.18%** |
| IndoBERT | Transformer | **87.81%** |
| SVM | TF-IDF + Classical ML | **83.20%** |
| Random Forest | TF-IDF + Classical ML | **77.91%** |

The transformer models are loaded using Hugging Face Transformers, while the SVM and Random Forest models use TF-IDF features and are downloaded from Hugging Face Hub.

---

## Project Structure

```text
Financial-Sentiment-Analyzer-main/
├── app.py
├── requirements.txt
├── README.md
├── Code Training Model/
│   ├── Classical_Train.ipynb
│   ├── train_IndoDistilBERT.ipynb
│   └── train_indobert.ipynb
└── Data Preparation & Evaluation/
    ├── Data_Preparation.ipynb
    └── Data_Prepare_&_Evaluation.ipynb
```

---

## Tech Stack

- **Python**
- **Streamlit**
- **PyTorch**
- **Hugging Face Transformers**
- **Hugging Face Hub**
- **Scikit-learn**
- **TF-IDF**
- **NumPy**
- **Joblib**

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Financial-Sentiment-Analyzer-main
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run the Streamlit application with:

```bash
streamlit run app.py
```

or:

```bash
python -m streamlit run app.py
```

Streamlit will display the local URL in the terminal, usually `http://localhost:8501`.

> The first startup may take longer because the trained models are downloaded from Hugging Face Hub and cached locally.

---

## Usage

1. Enter an Indonesian financial news headline in the text box.
2. Select **IndoDistilBERT, IndoBERT, SVM, or Random Forest**.
3. Click **Jalankan Analisis**.
4. View the predicted sentiment and model confidence.

Example input:

```text
IHSG menguat setelah investor merespons positif kebijakan ekonomi terbaru
```

---

## Model Resources

The application loads the trained models from Hugging Face Hub:

- `KevinRey/nlp-indobert-sentiment`
- `KevinRey/nlp-indodistilbert-sentiment`
- `KevinRey/nlp-classical-sentiment`

---

## Notes

The application is designed specifically for **Indonesian financial-news sentiment analysis**. Predictions should be treated as model outputs rather than financial advice or investment recommendations.
