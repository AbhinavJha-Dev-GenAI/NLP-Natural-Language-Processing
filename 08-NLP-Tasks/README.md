# 08. Core NLP Tasks 🎯

NLP is applied to solve various real-world problems. Most of these tasks can now be solved with high accuracy using Transformer models.

## 1. Text Classification 📊

Assigning a label to a piece of text.
- **Sentiment Analysis**: Positive, Negative, Neutral.
- **Spam Detection**: Spam vs. Not Spam.
- **Topic Modeling**: Categorizing news articles into "Sports", "Politics", etc.

---

## 2. Named Entity Recognition (NER) 👤📍

Identifying and categorizing key entities in text.
- *Entities:* Persons, Organizations, Locations, Dates.
- *Example:* "Apple [ORG] is meeting with Tim Cook [PER] in Cupertino [LOC] next Monday [DATE]."

---

## 3. Machine Translation 🌐

Translating text from one language to another.
- **Modern Approach**: Seq2Seq with Attention or Encoder-Decoder Transformers.
- **Evaluation**: **BLEU Score** (measures overlap between machine and human translation).

---

## 4. Text Summarization 📝

Condensing a long document into a shorter version.
- **Extractive**: Selecting the most important existing sentences from the text.
- **Abstractive**: Generating *new* sentences that capture the essence of the text (like a human would).

---

## 5. Question Answering (QA) ❓

- **Extractive QA**: Highlighting the answer within a provided context (e.g., SQuAD dataset).
- **Abstractive QA**: Generating an answer based on internal knowledge (e.g., ChatGPT).

---

## 🛠️ Essential Snippet (Hugging Face Pipeline)

```python
from transformers import pipeline

# Sentiment Analysis
classifier = pipeline("sentiment-analysis")
print(classifier("I love this NLP guide!"))

# Named Entity Recognition
ner = pipeline("ner", grouped_entities=True)
print(ner("Google is based in Mountain View, California."))

# Summarization
summarizer = pipeline("summarization")
print(summarizer("The Transformer is a deep learning model introduced in 2017... [long text]"))
```

---

## 📈 Evaluation Metrics
- **Accuracy/F1**: For Classification.
- **BLEU**: For Translation.
- **ROUGE**: For Summarization (focuses on recall).
