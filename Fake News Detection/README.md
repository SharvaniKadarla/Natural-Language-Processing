# Fake News Detection (Detect whether a news article is real or fake) using NLP and Machine Learning

## Overview
This project focuses on detecting whether a news article is **real or fake** using Natural Language Processing (NLP) techniques and classical Machine Learning models.

## Dataset
- Source: Kaggle Fake News Dataset  
- Link: https://www.kaggle.com/datasets/algord/fake-news  
- Size: ~23,000 news article titles  
- Labels: Real (1), Fake (0)

## Technologies Used
- Python
- NLP (Tokenization, Stopword Removal, Lemmatization)
- NLTK
- TF-IDF Vectorization
- Machine Learning (Naive Bayes, Logistic Regression, SVM)
- scikit-learn
- Matplotlib & Seaborn (Visualization)

## Project Workflow
1. Dataset exploration and class distribution analysis
2. Text preprocessing:
   - Lowercasing
   - URL removal
   - Special Characters, Punctuations and numbers removal
   - Tokenization
   - Stopword removal
   - Lemmatization or Stemming
3. Feature extraction using:
   - Bag of Words (CountVectorizer)
   - TF-IDF (Unigrams and Bigrams) / CountVectorizer
4. Model training:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Support Vector Machine (SVM)
5. Model evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
6. Visualization:
   - Accuracy comparison
   - PCA projection
   - t-SNE visualization
   - Cosine similarity analysis

## Results
- Logistic Regression and SVM achieved the best overall performance
- TF-IDF with bigrams and lemmatization improved classification accuracy
- SVM achieved the highest F1-score among all models

## How to Run
1. Clone the repository

2. Install dependencies:
`pip install -r requirements.txt`

3. Download the dataset directly from Kaggle or the dataset is available in the repository

4. Run the Jupyter or Colab Code Notebook

