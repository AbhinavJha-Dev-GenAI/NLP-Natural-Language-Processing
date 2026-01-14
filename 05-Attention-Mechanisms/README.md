# 05. Attention Mechanisms 🔍

Attention allows the model to "focus" on specific parts of the input sequence when producing each part of the output, rather than relying on a single fixed-length context vector.

## 1. Why Attention? 🏮

In traditional Seq2Seq models, the Encoder's last hidden state must represent the *entire* sentence. This becomes a bottleneck for long sequences. Attention allows the Decoder to "look back" at all Encoder hidden states.

---

## 2. Types of Attention 📏

### Bahdanau Attention (Additive)
One of the first attention mechanisms. It computes alignment scores using a small feed-forward network.

### Luong Attention (Multiplicative)
A more efficient version that uses a dot product between the Decoder's current state and the Encoder's states.

---

## 3. Self-Attention (The Core of Transformers) ⚡

Instead of attending to another sequence, the model attends to *itself* to understand relationships between words in the same sentence.

### The Query, Key, Value (Q, K, V) Intuition
- **Query ($Q$):** What I'm looking for.
- **Key ($K$):** What information I have to offer.
- **Value ($V$):** The actual content of the information.

**Score:** $Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

---

## 4. Multi-Head Attention 🎭

Using multiple attention mechanisms (heads) in parallel allows the model to capture different types of relationships simultaneously (e.g., one head for syntax, another for semantic meaning).

---

## 🛠️ Essential Snippet (Scaled Dot-Product Attention)

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    # Compute scores
    scores = torch.matmul(q, k.transpose(-2, -1)) /  torch.sqrt(torch.tensor(d_k))
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v), attn_weights
```

---

## 🧩 Key Takeaway
Attention effectively removes the "distance" between tokens, allowing the model to link words that are far apart in a sentence but semantically related (e.g., a pronoun and its referent).
