# 🧠 The NLP Pipeline Explorer

A premium, interactive educational tool to visualize the journey of human language into numerical logic. Discover how machines perceive text through cleaning, tokenization, normalization, and advanced embeddings.

![NLP Pipeline Banner](https://img.shields.io/badge/NLP-Explorer-indigo?style=for-the-badge&logo=natural-language-processing)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red?style=for-the-badge&logo=streamlit)
![Built with Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## ✨ Key Features

This application breaks down the NLP pipeline into 7 distinct, interactive steps:

1.  **🧼 Cleaning**: Remove noise like punctuation and special characters. 
2.  **✂️ Tokenization**: Split text into meaningful units (Words or Sentences).
3.  **📉 Normalization**: Apply Stopword removal, Stemming, or Lemmatization.
4.  **📚 Vocabulary**: See how the model builds its internal dictionary.
5.  **🔢 Vectorization**: Convert text to sparse vectors using **Bag-of-Words** or **TF-IDF**.
6.  **📍 Static Embeddings**: Visualize 300-dimension word vectors projected onto a 2D PCA plane.
7.  **🌌 Contextual Shift**: Experience how modern transformers like **BERT** change word embeddings based on surrounding context.

---

## 🛠️ Technology Stack

- **Core**: Python 3.9+ 
- **Interface**: [Streamlit](https://streamlit.io/) (Premium Light Modern Theme)
- **NLP Engine**: [NLTK](https://www.nltk.org/), [Scikit-learn](https://scikit-learn.org/)
- **Embeddings**: [Sentence-Transformers](https://sbert.net/) (Universal Sentence Encoder approach)
- **Visualization**: [Plotly Express](https://plotly.com/python/plotly-express/)
- **Data Handling**: Pandas, NumPy

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9 or higher and a virtual environment tool installed.

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-link>
cd nlp_pipeline_app

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the App
```bash
streamlit run app.py
```

---

## 💡 Usage Tip
The **Configuration Sidebar** on the left allows you to customize every aspect of the pipeline in real-time. Use the **Contextual Tab** to see the semantic difference between words that look the same but mean different things (e.g., *bank* as a river side vs. a financial bank).

---

## 🎨 UI/UX Philosophy
This app uses a **Premium Modern Light Theme** designed for maximum focus and readability. All visual elements are high-contrast and utilize a sophisticated indigo-slate palette perfect for educational demonstrations.

---

**Made with ❤️ for the NLP Community**
