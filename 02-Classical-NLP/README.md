# 02. Classical NLP & Vectorization 🏛️

Before deep learning, NLP relied on statistical methods to represent text as numerical vectors.

## 1. Bag of Words (BoW) 🎒

A simple way of representing text as a frequency count of words.
- **Binary BoW**: 1 if the word is present, 0 otherwise.
- **Count BoW**: The number of times the word appears in the document.
- *Issue:* It ignores word order and treats every word as equally important.

---

## 2. TF-IDF (Term Frequency - Inverse Document Frequency) ⚖️

A statistical measure used to evaluate how important a word is to a document in a collection (corpus).

*   **Term Frequency (TF)**: How often a word occurs in a document.
*   **Inverse Document Frequency (IDF)**: Decreases the weight of words that occur very frequently in the entire corpus (like "the", "a") and increases the weight of words that occur rarely.

**Formula:** $W_{x,y} = TF_{x,y} \times \log(\frac{N}{df_x})$

---

## 3. POS Tagging & Dependency Parsing 🏷️

*   **POS Tagging**: Assigning parts of speech (Noun, Verb, Adjective) to each word in a sentence.
*   **Dependency Parsing**: Analyzing the grammatical structure of a sentence to establish relationships between "head" words and words which modify those heads.

---

## 🛠️ Essential Snippet (Scikit-Learn TF-IDF)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# Print the feature names (vocabulary)
print(vectorizer.get_feature_names_out())

# Sparse matrix representation
print(X.toarray())
```

---

## ⚖️ When to use Classical NLP?
- For small datasets where deep learning might overfit.
- When you need high interpretability (you want to know *which* words led to a classification).
- For simple tasks like Spam detection or Sentiment analysis with high-frequency keywords.
