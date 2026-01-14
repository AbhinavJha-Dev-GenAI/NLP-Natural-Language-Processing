# 03. Word Embeddings 🌍

Word embeddings represent words as dense vectors in a high-dimensional space, where semantically similar words are positioned closer together.

## 1. Word2Vec (Google) 🧬

Learns embeddings by training a simple neural network to predict words based on their context.

*   **CBOW (Continuous Bag of Words)**: Predicts the center word from surrounding context words. (Fast, better for frequent words).
*   **Skip-gram**: Predicts the surrounding context words from a single center word. (Better for small datasets and rare words).

---

## 2. GloVe (Global Vectors - Stanford) 📊

Unlike Word2Vec which uses a local sliding window, GloVe uses **global co-occurrence statistics** of the entire corpus to learn word vectors.
- *Pros:* Combines the advantages of global matrix factorization (LSA) and local context window methods.

---

## 3. FastText (Facebook) 🚀

An extension of Word2Vec that treats each word as a bag of **character n-grams**.
- *Example:* "apple" -> `<ap`, `app`, `ppl`, `ple`, `le>`.
- *Big Advantage:* It can generate embeddings for **Out-of-Vocabulary (OOV)** words by summing up their character n-gram vectors.

---

## 4. Evaluating Embeddings 📏

*   **Cosine Similarity**: Measures the angle between two vectors. $\cos(\theta) = \frac{A \cdot B}{||A|| ||B||}$.
*   **Word Analogies**: Testing if $Vector(\text{King}) - Vector(\text{Man}) + Vector(\text{Woman}) \approx Vector(\text{Queen})$.

---

## 🛠️ Essential Snippet (Gensim Word2Vec)

```python
from gensim.models import Word2Vec

# Simple training
sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get vector
vector = model.wv['cat']

# Similarity
sim = model.wv.similarity('cat', 'dog')
print(f"Similarity: {sim}")
```

---

## ⚖️ Static vs. Contextual Embeddings
> [!NOTE]
> Word2Vec and GloVe are **Static**: the word "Bank" has the same vector in "river bank" and "bank account". Contextual embeddings (BERT) solve this by looking at surrounding words.
