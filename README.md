# Sentiment Analysis on IMDB Dataset using Word2Vec and Deep Learning

## 📌 Project Overview  
This project implements **sentiment analysis** on the [IMDB movie reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) using **Word2Vec embeddings** and **deep learning architectures**.  
The task is to classify reviews as **positive** or **negative**, leveraging semantic word representations to improve model performance.  

We evaluate and compare multiple deep learning models:
- **Convolutional Neural Networks (CNN)**
- **Long Short-Term Memory (LSTM)**
- **Bidirectional LSTM (Bi-LSTM)**
- **2-layer Bi-LSTM**
- **Hybrid CNN-LSTM**

The CNN model achieved the **highest accuracy of 89.02%**, demonstrating the effectiveness of convolutional filters in extracting local features from text.

---

## ✨ Features  
- Binary sentiment classification (Positive/Negative).  
- Preprocessing pipeline including HTML tag removal, stopword elimination, lemmatization, and text cleaning.  
- Word embeddings using **Google's pre-trained Word2Vec (300-dimensions)**.  
- Multiple deep learning architectures compared under the same setup.  
- Performance metrics: Accuracy, ROC curves, and Confusion matrices.  
- Implementation in **Python** with **TensorFlow/Keras**.

---

## 🛠️ Tech Stack  
- **Language:** Python  
- **Libraries/Frameworks:**  
  - `TensorFlow / Keras` – Model development  
  - `Gensim` – Word2Vec embeddings  
  - `NLTK` – Text preprocessing  
  - `NumPy`, `Pandas` – Data manipulation  
  - `Matplotlib`, `Seaborn` – Visualization  

---

## 📂 Dataset  
- **Name:** IMDB Movie Reviews Dataset  
- **Size:** 50,000 reviews (balanced: 25,000 positive + 25,000 negative)  
- **Split:** 80% Training (40,000) | 20% Testing (10,000)  

Dataset Source: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  

---

## 🔧 Preprocessing Steps  
1. **Fix contractions** (e.g., "don't" → "do not").  
2. **Remove HTML tags** (e.g., `<br>`, `<i>`).  
3. **Convert to lowercase**.  
4. **Remove special characters and punctuation**.  
5. **Stopword removal** using NLTK.  
6. **Lemmatization** (e.g., "running" → "run").  
7. **Standardization:** Truncate/pad reviews to a fixed length of 300 tokens.

---

## 🔑 Word Embeddings  
- Pre-trained **Word2Vec (Google News, 300D)** embeddings via `Gensim`.  
- Out-of-vocabulary (OOV) words mapped to zero vectors.  

---

## 🏗️ Model Architectures  
1. **CNN (Convolutional Neural Network)**  
   - Best performance (Accuracy: **89.02%**, AUC: 0.96).  
   - Captures local n-gram-like features effectively.  

2. **LSTM (Long Short-Term Memory)**  
   - Accuracy: **88.05%**  
   - Effective in modeling sequential dependencies.  

3. **Bi-LSTM (Bidirectional LSTM)**  
   - Accuracy: **87.63%**  
   - Processes sequences in both forward and backward directions.  

4. **2-layer Bi-LSTM**  
   - Accuracy: **88.52%**  
   - Deeper recurrent architecture improves feature extraction.  

5. **Hybrid CNN-LSTM**  
   - Accuracy: **88.30%**  
   - Combines CNN for local features and LSTM for sequential context.  

---

## 📊 Results  

| Model            | Accuracy (%) |
|-------------------|--------------|
| CNN              | **89.02**    |
| Bi-LSTM (2-layer)| 88.52        |
| CNN-LSTM Hybrid  | 88.30        |
| LSTM             | 88.05        |
| Bi-LSTM          | 87.63        |

- **CNN performed best** in this task, highlighting its strength in capturing local patterns.

---

## 📈 Visualizations  
- Confusion matrices for all models.  
- ROC curves (AUC ~0.95 for most models).  
- Model accuracy comparison bar chart.

---

## 🚀 Future Work  
- Incorporate transformer-based models like **BERT** and **RoBERTa**.  
- Expand sentiment classes beyond binary (add neutral/mixed).  
- Explore attention mechanisms for better interpretability.  
- Apply multi-lingual sentiment analysis.  
- Optimize models for **real-time sentiment analysis** in social media streams.  

---

## 📚 References  
- [Word2Vec - Google News (300D)](https://code.google.com/archive/p/word2vec/)  
- Deep learning and NLP research papers cited in the report.

---

## 🧑‍💻 Authors  
- **Shubham Soni** – [LinkedIn](https://www.linkedin.com/in/shubham-soni05/) | shubham134soni@gmail.com  
- **Dr. Natesha B V** – IIIT Raichur  

---

## 🏷️ Keywords  
`Sentiment Analysis` `IMDB Dataset` `Word2Vec` `Deep Learning` `CNN` `LSTM` `Bi-LSTM` `Text Classification` `NLP`  

---
