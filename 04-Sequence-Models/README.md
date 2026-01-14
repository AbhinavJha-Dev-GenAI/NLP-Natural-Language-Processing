# 04. Sequence Models (RNNs & LSTMs) 🔄

For tasks like translation or sentiment analysis, the order of words matters. Sequence models process text as a stream of tokens.

## 1. Recurrent Neural Networks (RNN) 🌀

RNNs have a "hidden state" that acts as memory, allowing information from past time steps to influence the current prediction.
- **Problem**: **Vanishing Gradient**. As the sequence gets longer, the model "forgets" information from the early steps.

---

## 2. LSTMs & GRUs 🧪

Designed to solve the vanishing gradient problem using **Gates**.

*   **LSTM (Long Short-Term Memory)**: Uses Input, Forget, and Output gates to carefully control a "Cell State" (long-term memory).
*   **GRU (Gated Recurrent Unit)**: A simplified version with fewer gates (Update and Reset), often performing similar to LSTM but faster.

---

## 3. Bidirectional RNNs ↔️

Processes the sequence from left-to-right *and* right-to-left simultaneously. This allows the model to have context from both the past and the future for every word.

---

## 4. Seq2Seq (Encoder-Decoder) 🗺️

A framework where one RNN (Encoder) processes the input sequence into a "Context Vector," and another RNN (Decoder) generates the output sequence from that vector.
- *Use Case:* Machine Translation, Summarization.

---

## 🛠️ Essential Snippet (PyTorch LSTM)

```python
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        # output: all hidden states, (hn, cn): last state
        output, (hn, cn) = self.lstm(embedded)
        return self.fc(hn[-1]) # Use the last hidden state for classification
```

---

## 📊 The Bottleneck
The "Context Vector" in Seq2Seq is a fixed-size vector. Compressing a long sentence into a single vector leads to information loss. This is why **Attention** was invented.
