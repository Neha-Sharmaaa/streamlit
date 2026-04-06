import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import (
    clean_text, tokenize, stopword_and_norm, get_vocabulary,
    get_vectorization, get_static_embeddings, get_contextual_embeddings
)

# Set Page Config
st.set_page_config(
    page_title="Transformer Journey: Text to Embeddings",
    page_icon="🧠",
    layout="wide",
)

# Load CSS
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# App Title & Intro
st.title("🧠 The NLP Pipeline Explorer")
st.markdown("""
### Discover how machines turn human language into numerical logic.
Follow the journey of a sentence through preprocessing, vectorization, and embeddings.
""")

# Sidebar Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Theme Sync for Charts
    theme_mode = st.radio("App Theme Mode", ["Light", "Dark"], horizontal=True)
    chart_color = "#1e293b" if theme_mode == "Light" else "#f8fafc"
    
    if theme_mode == "Dark":
        st.markdown("""
            <style>
                :root {
                    --primary: #818cf8;
                    --primary-hover: #6366f1;
                    --background: #0f172a;
                    --secondary: #1e293b;
                    --accent: #fb7185;
                    --text-primary: #f8fafc;
                    --text-secondary: #94a3b8;
                    --glass: rgba(30, 41, 59, 1);
                    --glass-border: rgba(51, 65, 85, 0.5);
                    --token-bg: #312e81;
                    --token-border: #4338ca;
                    --card-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
                }
                .stApp { background-color: var(--background) !important; }
            </style>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    user_input = st.text_area(
        "Enter raw text here:",
        "The quick brown fox jumps over the lazy dog. NLP is amazing for data science!",
        height=150
    )
    
    st.divider()
    
    st.subheader("🧹 Cleaning Options")
    lowercasing = st.toggle("Lowercasing", True)
    remove_punct = st.toggle("Remove Punctuation", True)
    remove_special = st.toggle("Remove Special Characters", False)
    
    cleaning_options = {
        'Lowercase': lowercasing,
        'Remove Punctuation': remove_punct,
        'Remove Special Characters': remove_special
    }

    st.divider()
    
    st.subheader("✂️ Tokenization")
    token_type = st.selectbox(
        "Tokenization Level",
        ["word", "subword", "character"]
    )
    
    st.divider()
    
    st.subheader("📊 Normalization")
    remove_stop = st.toggle("Remove Stopwords", True)
    norm_method = st.radio(
        "Stemming / Lemmatization",
        ["None", "Stemming", "Lemmatization"],
        index=2
    )
    
    norm_options = {
        'Remove Stopwords': remove_stop,
        'Normalization': norm_method
    }
    
    st.divider()
    
    st.subheader("🔢 Vectorization")
    vector_method = st.selectbox(
        "Method",
        ["Bag-of-Words", "TF-IDF"]
    )

# --- PROCESSING FLOW ---
cleaned = clean_text(user_input, cleaning_options)
tokens = tokenize(cleaned, token_type)
norm_tokens = stopword_and_norm(tokens, norm_options)
vocab_df = get_vocabulary(norm_tokens)

# --- MAIN UI TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🧹 Cleaning", "✂️ Tokenization", "🗜️ Normalization", 
    "📚 Vocabulary", "🔢 Vectorization", "🗺️ Static Embeddings", "🤖 Contextual"
])

# TAB 1: Cleaning
with tab1:
    st.markdown('<div class="nlp-card"><div class="nlp-step-title">1. Text Cleaning & Normalization</div>'
                '<div class="nlp-explanation">Cleaning removes noise and standardizes text to make it consistent for the model. </div></div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Text")
        st.info(user_input)
    
    with col2:
        st.subheader("Cleaned Text")
        st.success(cleaned)
        
    st.markdown("---")
    st.write("### Comparison")
    # Show diff simple logic
    words_orig = user_input.split()
    words_clean = cleaned.split()
    
    # Simple mockup of showing removed logic
    st.write("Removing punctuation and standardizing cases helps the model treat 'Apple' and 'apple' as the same thing.")

# TAB 2: Tokenization
with tab2:
    st.markdown('<div class="nlp-card"><div class="nlp-step-title">2. Tokenization</div>'
                '<div class="nlp-explanation">Breaking text into smaller pieces called tokens. This can be at the word, subword, or character level.</div></div>', 
                unsafe_allow_html=True)
    
    st.subheader(f"Current Level: {token_type.capitalize()}")
    
    cols = st.columns(len(tokens)//10 + 1)
    
    st.write("### Resulting Tokens:")
    html_tokens = "".join([f'<span class="token-chip">{t}</span>' for t in tokens])
    st.markdown(html_tokens, unsafe_allow_html=True)
    
    st.markdown("---")
    st.write("**Why subwords?** Models like BERT use subwords to handle new words by breaking them into familiar pieces (e.g., 'unbelievable' -> 'un', 'believ', 'able').")

# TAB 3: Normalization
with tab3:
    st.markdown('<div class="nlp-card"><div class="nlp-step-title">3. Stopword Removal & Normalization</div>'
                '<div class="nlp-explanation">Reduces words to their core meaning and removes fluff words (like "the", "a") that don\'t add semantic value.</div></div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Before (Tokens)")
        st.write(tokens[:15])
        
    with col2:
        st.write(f"### After ({norm_method})")
        st.write(norm_tokens[:15])
        
    st.markdown("---")
    st.write("**Stemming** chops off suffixes (run, running -> run). **Lemmatization** uses a dictionary to find the root word (was, am -> be).")

# TAB 4: Vocabulary building
with tab4:
    st.markdown('<div class="nlp-card"><div class="nlp-step-title">4. Building the Vocabulary</div>'
                '<div class="nlp-explanation">The model builds a dictionary of all unique words it has seen. Every word gets a unique "ID".</div></div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Token -> Index Mapping")
        st.dataframe(vocab_df[['Token', 'Index']].set_index('Index'), height=400)
    
    with col2:
        st.subheader("Frequency Metrics")
        fig = px.bar(vocab_df.sort_values('Frequency', ascending=False).head(15), 
                     x='Token', y='Frequency', 
                     color='Frequency', color_continuous_scale='Viridis')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=chart_color)
        st.plotly_chart(fig, use_container_width=True)

# TAB 5: Vectorization
with tab5:
    st.markdown('<div class="nlp-card"><div class="nlp-step-title">5. Vectorization (Numbers!)</div>'
                '<div class="nlp-explanation">Converting tokens into fixed-size numerical vectors. <b>Bag-of-Words</b> counts occurrences, <b>TF-IDF</b> weighs importance.</div></div>', 
                unsafe_allow_html=True)
    
    vector_df = get_vectorization([cleaned], vector_method)
    
    st.subheader(f"Sentence Vector ({vector_method})")
    st.dataframe(vector_df, height=150)
    
    # Heatmap
    fig = px.imshow(vector_df, 
                    labels=dict(x="Token", y="Sentence Index", color="Weight"),
                    x=vector_df.columns,
                    aspect="auto",
                    color_continuous_scale='magma')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color=chart_color)
    st.plotly_chart(fig, use_container_width=True)

# TAB 6: Static Embeddings
with tab6:
    st.markdown('<div class="nlp-card"><div class="nlp-step-title">6. Static Word Embeddings</div>'
                '<div class="nlp-explanation">Words are mapped to dense vectors where similar meanings are close together in space. </div></div>', 
                unsafe_allow_html=True)
    
    embedding_df = get_static_embeddings(norm_tokens)
    
    fig = px.scatter(embedding_df, x='x', y='y', text='word', size_max=60)
    fig.update_traces(textposition='top center', marker=dict(size=12, color='#6366f1'))
    fig.update_layout(
        title="2D Projection of Word Vectors (PCA)",
        xaxis_title="Semantic Dimension 1",
        yaxis_title="Semantic Dimension 2",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color=chart_color
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 Notice how words used in similar contexts tend to drift toward each other in this 2D space.")

# TAB 7: Contextual Embeddings
with tab7:
    st.markdown('<div class="nlp-card"><div class="nlp-step-title">7. Contextual Embeddings (BERT style)</div>'
                '<div class="nlp-explanation">The same word can have different vectors depending on the surrounding words.</div></div>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        s1 = st.text_input("Sentence 1", "The river bank is overflowing.")
    with col2:
        s2 = st.text_input("Sentence 2", "I went to the bank to deposit cash.")
        
    target = st.text_input("Target Word to Compare", "bank")
    
    if st.button("Analyze Context"):
        with st.spinner("Calculating semantic shift..."):
            components = get_contextual_embeddings(s1, s2, target)
            
            df_context = pd.DataFrame({
                'x': components[:, 0],
                'y': components[:, 1],
                'Context': [f'Sentence 1: "{s1}"', f'Sentence 2: "{s2}"'],
                'Word': [target, target]
            })
            
            fig = px.scatter(df_context, x='x', y='y', color='Context', text='Word', title=f'Contextual Drift for "{target}"')
            fig.update_traces(textposition='top center', marker=dict(size=20))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color=chart_color)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"In modern models like BERT, the word **'{target}'** has a different numerical home depending on whether it's a river side or a financial institution!")

st.markdown("---")
st.markdown("Made with ❤️ for NLP Enthusiasts")
