```python
import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

# Download the required NLTK data
nltk.download('punkt')

# Load the dataset with caching to improve performance
@st.cache_data
def load_data():
    data = pd.read_csv('Ecommerce_product.csv')
    data = data.drop('id', axis=1)
    return data

# Define tokenizer and stemmer
stemmer = SnowballStemmer('english')
def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Create stemmed tokens column
def create_stemmed_tokens_column(data):
    data['stemmed_tokens'] = data.apply(
        lambda row: tokenize_and_stem(str(row['Title']) + ' ' + str(row['Description'])), axis=1
    )
    return data

# TF-IDF vectorizer and cosine similarity function
tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, token_pattern=None)

def cosine_sim(text1, text2):
    text1_concatenated = ' '.join(text1)
    text2_concatenated = ' '.join(text2)
    tfidf_matrix = tfidf_vectorizer.fit_transform([text1_concatenated, text2_concatenated])
    return cosine_similarity(tfidf_matrix)[0][1]

# Search function for products
def search_products(query, data):
    query_stemmed = tokenize_and_stem(query)
    data['similarity'] = data['stemmed_tokens'].apply(lambda x: cosine_sim(query_stemmed, x))
    results = data.sort_values(by=['similarity'], ascending=False).head(10)
    if results.empty:
        return None
    return results[['Title', 'Description', 'Category', 'similarity']]

# Main function to run the Streamlit app
def main():
    # Load the image and display it
    img = Image.open('swift.png')
    st.image(img, use_column_width=True)

    st.markdown(
        "<h1 style='text-align: center; color: #ff5733;'>Search Engine & Product Recommendation</h1>",
        unsafe_allow_html=True
    )

    # Load and preprocess the data
    data = load_data()
    data = create_stemmed_tokens_column(data)

    # Custom CSS for styling
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #ff5733;
                color: white;
                padding: 10px 25px;
                border-radius: 8px;
                border: none;
                font-size: 16px;
                font-weight: bold;
                transition: 0.3s;
            }
            .stButton>button:hover {
                background-color: #e04b20;
                transform: scale(1.05);
            }
            .search-card {
                background: #ffffff;
                padding: 15px;
                border-radius: 12px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                margin-bottom: 15px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Input layout in columns
    col1, col2 = st.columns([4,1])
    with col1:
        query = st.text_input("Enter Product Name", key="query", max_chars=100, placeholder="Search by product or category...")
    with col2:
        submit = st.button('Search', key='search_button')

    if submit:
        if query:
            with st.spinner("🔎 Searching for best matches..."):
                res = search_products(query, data)

            if res is not None:
                st.subheader("Top Matching Products:")
                for _, row in res.iterrows():
                    st.markdown(
                        f"""
                        <div class="search-card">
                            <h4 style="color:#ff5733;">{row['Title']}</h4>
                            <p><b>Category:</b> {row['Category']}</p>
                            <p>{row['Description'][:200]}...</p>
                            <p style="font-size:13px; color:grey;">Similarity Score: {row['similarity']:.2f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.error("❌ No matching products found. Try a different query.")
        else:
            st.warning("⚠️ Please enter a product name to search.")

# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()
```
