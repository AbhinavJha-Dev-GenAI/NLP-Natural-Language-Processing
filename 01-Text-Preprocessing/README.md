# 01. Text Preprocessing 🧹

Text preprocessing is the crucial first step in any NLP pipeline. Raw text is messy, and models perform better when the data is clean and consistently formatted.

## 1. Basic Cleaning 🧼

*   **Lowercasing**: Reducing the vocabulary size by treating "Apple" and "apple" as the same.
*   **Punctuation Removal**: Removing symbols like `!`, `?`, `.`, as they usually don't carry semantic meaning for many tasks (though they might for sentiment or parsing).
*   **Stopword Removal**: Removing frequent words like "the", "is", "at" which often add noise without adding significant value.
*   **Noise Handling**: Using Regex to strip out HTML tags, URLs, and extra whitespaces.

---

## 2. Normalization: Stemming vs. Lemmatization ⚖️

Both techniques aim to reduce a word to its base or root form.

### Stemming
A heuristic process that chops off the ends of words.
- *Example:* "Running" -> "Run", "Better" -> "Bet".
- *Pros:* Fast, simple. 
- *Cons:* Can result in non-words (Over-stemming).

### Lemmatization
Uses a vocabulary and morphological analysis (WordNet) to return the dictionary form of a word.
- *Example:* "Running" -> "Run", "Better" -> "Good".
- *Pros:* Extremely accurate, linguistically correct.
- *Cons:* Slower, requires part-of-speech context.

---

## 3. Segmentation (Tokenization) ✂️

Breaking down a large chunk of text into smaller units (tokens) like sentences or words.

*   **Sentence Tokenization**: Splitting a paragraph into sentences.
*   **Word Tokenization**: Splitting a sentence into individual words.
*   **N-grams**: Generating contiguous sequences of $n$ items from a given sample of text.
    - *Bigram:* "Natural Language", "Language Processing".

---

## 🛠️ Essential Snippet (NLTK & SpaCy)

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

# NLTK Lemmatization
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("better", pos="a")) # Returns "good"

# SpaCy Preprocessing (Modern Way)
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for token in doc:
    print(f"{token.text} -> Lemma: {token.lemma_}, POS: {token.pos_}")
```

---

## 🧪 Best Practices
1.  **Don't over-clean**: For tasks like BERT-based classification, keeping case and punctuation can actually help.
2.  **Domain Matters**: Medical or Legal text preprocessing requires specialized stopword lists and lemmatizers.
