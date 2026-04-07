# <img src="https://api.iconify.design/material-symbols:psychology.svg?color=%234f46e5" width="32" align="center"> The NLP Pipeline Explorer

A premium, interactive educational tool to visualize the journey of human language into numerical logic. Discover how machines perceive text through cleaning, tokenization, normalization, and advanced embeddings.

![NLP Pipeline Banner](https://img.shields.io/badge/NLP-Explorer-indigo?style=for-the-badge&logo=natural-language-processing)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-red?style=for-the-badge&logo=streamlit)
![Built with Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## <img src="https://api.iconify.design/material-symbols:star-sparkles-outline.svg?color=%234f46e5" width="24" align="center"> Key Features

This application breaks down the NLP pipeline into 7 distinct, interactive steps:

1.  **<img src="https://api.iconify.design/material-symbols:cleaning-services.svg?color=%234f46e5" width="20" align="center"> Cleaning**: Remove noise like punctuation and special characters. 
2.  **<img src="https://api.iconify.design/material-symbols:content-cut.svg?color=%234f46e5" width="20" align="center"> Tokenization**: Split text into meaningful units (Words or Sentences).
3.  **<img src="https://api.iconify.design/material-symbols:bar-chart.svg?color=%234f46e5" width="20" align="center"> Normalization**: Apply Stopword removal, Stemming, or Lemmatization.
4.  **<img src="https://api.iconify.design/material-symbols:library-books.svg?color=%234f46e5" width="20" align="center"> Vocabulary**: See how the model builds its internal dictionary.
5.  **<img src="https://api.iconify.design/material-symbols:format-list-numbered.svg?color=%234f46e5" width="20" align="center"> Vectorization**: Convert text to sparse vectors using **Bag-of-Words** or **TF-IDF**.
6.  **<img src="https://api.iconify.design/material-symbols:scatter-plot.svg?color=%234f46e5" width="20" align="center"> Static Embeddings**: Visualize 300-dimension word vectors projected onto a 2D PCA plane.
7.  **<img src="https://api.iconify.design/material-symbols:explore.svg?color=%234f46e5" width="20" align="center"> Contextual Shift**: Experience how modern transformers like **BERT** change word embeddings based on surrounding context.

---

## <img src="https://api.iconify.design/material-symbols:build.svg?color=%234f46e5" width="24" align="center"> Technology Stack

- **Core**: Python 3.9+ 
- **Interface**: [Streamlit](https://streamlit.io/) (Premium Light Modern Theme)
- **NLP Engine**: [NLTK](https://www.nltk.org/), [Scikit-learn](https://scikit-learn.org/)
- **Embeddings**: [Sentence-Transformers](https://sbert.net/) (Universal Sentence Encoder approach)
- **Visualization**: [Plotly Express](https://plotly.com/python/plotly-express/)
- **Data Handling**: Pandas, NumPy

---

## <img src="https://api.iconify.design/material-symbols:rocket-launch.svg?color=%234f46e5" width="24" align="center"> Getting Started

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

## <img src="https://api.iconify.design/material-symbols:lightbulb.svg?color=%234f46e5" width="24" align="center"> Usage Tip
The **Configuration Sidebar** on the left allows you to customize every aspect of the pipeline in real-time. Use the **Contextual Tab** to see the semantic difference between words that look the same but mean different things (e.g., *bank* as a river side vs. a financial bank).

---

## <img src="https://api.iconify.design/material-symbols:palette.svg?color=%234f46e5" width="24" align="center"> UI/UX Philosophy
This app uses a **Premium Modern Light Theme** designed for maximum focus and readability. All visual elements are high-contrast and utilize a sophisticated indigo-slate palette perfect for educational demonstrations.

---

**Made with <img src="https://api.iconify.design/material-symbols:favorite.svg?color=%23e11d48" width="20" align="center"> for the NLP Community**
