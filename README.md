# **Live Demo**
🚀 Explore the deployed **Amazon Product Search Engine** here:  
🔗 [Search Recommender SwiftBuy](https://searchrecommenderswiftbuy.streamlit.app/)

---

# **SwiftBuy Product Search Engine**

This project implements a search engine for products in the **Amazon Product Dataset** using **Natural Language Processing (NLP)** techniques and a **Streamlit-based web application**.

## **Features**
- 🔍 **Search Functionality**: Users can search for products by entering a query.  
- 🧠 **NLP Processing**: Utilizes **tokenization, stemming, and TF-IDF vectorization** to preprocess and analyze product titles and descriptions.  
- 📊 **Relevance Ranking**: Returns the **top 10 most relevant products** based on **cosine similarity** between the query and dataset.  
- 🎨 **User-Friendly Interface**: Built using **Streamlit**, allowing for easy interaction and visualization of results.  

---

## **Dataset**
- 📂 **Name**: Amazon Product Dataset  
- 🌐 **Source**: [Kaggle](https://www.kaggle.com/)  
- 📖 **Description**: Contains information about over **1.3 million products**, including titles, descriptions, categories, and prices.  
- 🔢 **Columns Used**: `Title`, `Description`, and `Category`  

---

## **Installation**
Follow these steps to set up and run the project locally:  

### **Prerequisites**
- 🐍 **Python 3.8 or higher**  
- Install the required dependencies using:  
  ```bash
  pip install -r requirements.txt
  ```

### **Run the App**
```bash
streamlit run app.py

