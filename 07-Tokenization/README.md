# 07. Tokenization Methods ✂️

Tokenization is the process of breaking text into subword units. Modern NLP models use subword tokenization to handle large vocabularies and rare words efficiently.

## 1. Why Subword Tokenization? 🧩

*   **Word-level**: Too many words (millions). OOV (Out-of-Vocabulary) problem.
*   **Character-level**: Sequences become too long. Loses semantic meaning of words.
*   **Subword**: Balances vocabulary size and sequence length. Can represent any word by breaking it into pieces.

---

## 2. Common Algorithms 📜

### BPE (Byte Pair Encoding)
Used by **GPT, RoBERTa, Llama**.
- Starts with characters and iteratively merges the most frequent pairs of characters/sequences into a single token.
- *Example:* "low" and "lower" might share the subword "low".

### WordPiece
Used by **BERT, DistilBERT**.
- Similar to BPE, but instead of merging based on frequency, it merges based on the likelihood of the training data using a language model.

### SentencePiece (Unigram)
Used by **T5, ALBERT, Llama (sometimes)**.
- Treats the input as a raw stream of bytes (including spaces). This makes it language-independent (no need for a pre-tokenizer).

---

## 3. Special Tokens 🔑

Modern tokenizers use special markers to guide the model:
- `[CLS]` / `<s>`: Beginning of a sequence (often used for classification).
- `[SEP]` / `</s>`: Separator between two sequences.
- `[PAD]`: Padding to make sequences the same length.
- `[MASK]`: Masked token for training (MML).

---

## 🛠️ Essential Snippet (Hugging Face Tokenizers)

```python
from transformers import AutoTokenizer

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Tokenization is awesome!"
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)

print(f"Tokens: {tokens}")
print(f"IDs: {ids}")

# Decoding
print(tokenizer.decode(ids))
```

---

## 🧪 Comparison Table

| Algorithm | Model | Handling of Spaces |
| :--- | :--- | :--- |
| **BPE** | GPT-2 | Pre-tokenizer required |
| **WordPiece** | BERT | Uses `##` prefix for subwords |
| **SentencePiece** | T5 | Uses `_` to represent spaces |
