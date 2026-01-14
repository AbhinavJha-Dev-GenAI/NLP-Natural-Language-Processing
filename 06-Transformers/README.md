# 06. The Transformer Architecture 🏛️

The Transformer architecture, introduced in "Attention Is All You Need" (2017), replaced recurrence with self-attention, enabling massive parallelization and state-of-the-art performance.

## 1. High-Level Architecture 🏗️

A Transformer consists of an **Encoder** and a **Decoder**. Unlike RNNs, it processes the entire sequence at once.

### Key Components:
*   **Positional Encoding**: Since there's no recurrence, we add sinusoidal signals to embeddings to let the model know the position of each word.
*   **Multi-Head Self-Attention**: Extracted relationships between all words in the sequence.
*   **Residual Connections & Layer Norm**: Essential for training very deep networks without gradient issues.

---

## 2. Encoder Block 🧱

The Encoder consists of a stack of identical layers. Each layer has:
1.  **Multi-Head Self-Attention**.
2.  **Position-wise Feed-Forward Network (FFN)**.
- *Goal:* Create a rich, contextual representation of the input.

---

## 3. Decoder Block 🧱

Similar to the Encoder but with two major differences:
1.  **Masked Multi-Head Attention**: Prevents the model from "peeking" at future tokens during training.
2.  **Encoder-Decoder Attention**: Allows the Decoder to focus on relevant parts of the Encoder's output.

---

## 4. Why is it so powerful? ⚡

1.  **Parallelization**: No more sequential processing (tokens are processed in parallel).
2.  **Global Receptive Field**: Any token can "talk" to any other token directly in one layer.
3.  **Scalability**: Transformers scale incredibly well with more data and more parameters (leading to LLMs).

---

## 🛠️ Essential Snippet (Transformer in PyTorch)

```python
import torch.nn as nn

# PyTorch provides a highly optimized Transformer module
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# Example input: (seq_len, batch, d_model)
src = torch.randn(10, 32, 512)
out = transformer_encoder(src)
```

---

## 🌍 The Legacy
The Transformer architecture is the foundation for almost all modern NLP models:
- **BERT** (Encoder-only)
- **GPT** (Decoder-only)
- **T5 / BART** (Encoder-Decoder)
