# 13. NLP Interview Preparation 🧠

Common technical and behavioral questions for NLP/ML roles.

## 1. Core Concepts 📖

*   **Q: Why do we use padding in batch training?**
    - *A:* Neural networks require inputs in a batch to have the same dimensions. Padding adds dummy tokens (usually zeros) to shorter sentences so they match the length of the longest sentence in the batch.
*   **Q: What is the "Out-of-Vocabulary" (OOV) problem and how do subword tokenizers solve it?**
    - *A:* OOV occurs when a model encounters a word it hasn't seen during training. Subword tokenizers (like BPE) break unknown words into smaller known units (like "un" + "known"), ensuring every word can be represented.
*   **Q: Explain the difference between BLEU and ROUGE.**
    - *A:* **BLEU** focuses on precision (how much of the machine translation is in the human reference). **ROUGE** focuses on recall (how much of the human reference is captured by the machine summary).

---

## 2. Advanced NLP & LLMs 🧪

*   **Q: Why is Positional Encoding necessary in Transformers?**
    - *A:* Transformers process all tokens in parallel and lack an inherent sense of order (unlike RNNs). Positional encoding adds "time" or "position" information to the word embeddings.
*   **Q: What is the "Attention Bottleneck"?**
    - *A:* Traditional RNN-based Seq2Seq models compress the whole input into one vector. Attention allows the decoder to selectively look at any input token, solving the bottleneck.
*   **Q: What is LoRA fine-tuning?**
    - *A:* It's a method where we only train a small number of added parameters (low-rank matrices) while keeping the original model weights frozen, making fine-tuning much faster and memory-efficient.

---

## 3. Practical Scenarios 🛠️

*   **Scenario: Your model is biased against a certain demographic.**
    - *A:* Use bias detection tools like **AI Fairness 360**, audit your training data for imbalances, and consider "Debiased" embeddings or counter-factual data augmentation.
*   **Scenario: How to handle extremely long documents (>2048 tokens) with BERT?**
    - *A:* BERT has a token limit (usually 512). Solutions include: **Chunking** (splitting and averaging), **Longformer**, or using models designed for long sequences like **BigBird**.

---

## 🎯 NLP Cheat Sheet
1. **Low Data**: Classical NLP (TF-IDF + SVM).
2. **Standard Task**: BERT/RoBERTa Fine-tuning.
3. **Generative Task**: GPT/T5.
4. **Semantic Search**: Sentence-Transformers + FAISS.
