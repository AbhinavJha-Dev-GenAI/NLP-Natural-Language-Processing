# 10. Advanced & Contextual Embeddings 🧠

The shift from static vectors (Word2Vec) to contextual representations (BERT) revolutionized NLP.

## 1. Contextual Embeddings 🎭

Unlike static embeddings, contextual embeddings change based on the surrounding words.
- *Example:* The word "Bank" in "Bank of the river" and "Bank of England" will have different vectors.

### Key Models:
*   **ELMo**: Uses bidirectional LSTMs to create contextual vectors.
*   **BERT**: Uses bidirectional Transformers to look at the entire sentence at once.
*   **RoBERTa**: An optimized version of BERT trained on more data for longer.

---

## 2. Sentence Embeddings 📝

Sometimes we need a vector for an entire sentence or paragraph (e.g., for semantic search).

*   **S-BERT (Sentence-BERT)**: A modification of BERT using Siamese networks to produce semantically meaningful sentence embeddings that can be compared using cosine similarity.
*   **Cross-Encoders**: More accurate but much slower; they process two sentences together to get a similarity score.

---

## 3. Vector Databases & Similarity Search 🔍

Once you have embeddings, you need to store and search them efficiently.

*   **FAISS (Facebook AI Similarity Search)**: A library for efficient similarity search and clustering of dense vectors.
*   **Vector DBs**: Tools like **Chroma, Pinecone, or Weaviate** that manage embeddings, metadata, and provide fast retrieval for RAG (Retrieval-Augmented Generation) systems.

---

## 🛠️ Essential Snippet (Sentence Transformers)

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = ["The cat sits outside", "A man is playing guitar"]
embeddings = model.encode(sentences)

# Compute cosine similarity
cos_sim = util.cos_sim(embeddings[0], embeddings[1])
print(f"Similarity: {cos_sim}")
```

---

## 🧩 Practical Use Case
Contextual embeddings are the engine behind modern **Search Engines** (Google uses BERT) and **Recommendation Systems**, where understanding the intent behind a query is more important than matching keywords.
