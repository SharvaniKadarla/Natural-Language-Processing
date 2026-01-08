# Sentiment Analysis: Classical ML (Machine Learning) vs BERT (Bidirectional Encoder Representations from Transformers)

## Project Overview
This project performs sentiment analysis on three types of real-world text data:
- Movie reviews
- Product reviews
- Tweets

Two sentiment classification approaches are implemented and compared:
1. Classical machine learning models using TF-IDF features
2. Transformer-based sentiment classification using BERT

   The goal is to analyze performance differences between traditional feature-based models and modern transformer models using the same datasets and evaluation metrics.

---

## Datasets Used
- **IMDB Movie Reviews** (Hugging Face)
- **Amazon Polarity Product Reviews** (Hugging Face)
- **Sentiment140 Twitter Dataset** (Kaggle)

A balanced subset of 100 samples per dataset was used.

---

## Text Preprocessing (Classical Models)
- Lowercasing
- Tokenization (NLTK)
- Stopword removal
- Lemmatization
- Removal of non-alphabetic tokens

Preprocessing was applied only to classical ML models.

---

## Models Implemented

### Classical Machine Learning
- Logistic Regression
- Multinomial Naive Bayes
- Feature Engineering: TF-IDF Vectorization

### Transformer-Based Model
- DistilBERT (fine-tuned on SST-2)
- Implemented using Hugging Face Transformers pipeline
- No manual feature engineering required

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

Models were evaluated on identical test sets for fair comparison.

---

## Key Observations
- Classical models show high precision but low recall on longer or nuanced text.
- Naive Bayes performs slightly better on short, informal text (tweets).
- BERT consistently outperforms classical models across all datasets by capturing
  contextual meaning.

---

## Technologies Used
- Python
- NLTK
- scikit-learn
- Hugging Face Transformers
- Hugging Face Datasets
- Pandas, NumPy
- Jupyter Notebook

---

## How to Run
1. Install dependencies:
`pip install datasets transformers scikit-learn pandas nltk`

2. Open the notebook and enter the code

3. Run all cells sequentially.

