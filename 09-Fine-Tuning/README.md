# 09. Fine-Tuning NLP Models 🛠️

Fine-tuning is the process of taking a pre-trained model (trained on massive data like Wikipedia) and training it further on a smaller, task-specific dataset.

## 1. Why Fine-Tune? 🚀

*   **Knowledge Transfer**: Models like BERT already understand grammar, syntax, and some world knowledge. Fine-tuning "teaches" them your specific labels.
*   **Efficiency**: Training from scratch requires thousands of GPUs and months of time. Fine-tuning takes minutes to hours on a single GPU.

---

## 2. Fine-Tuning Strategies 🎯

### Full Fine-Tuning
Update all parameters of the pre-trained model. Most accurate but computationally expensive.

### Feature Extraction (Freezing)
Freeze the Transformer layers and only train a new classification "head" on top. Fast and efficient if the pre-trained features are already relevant.

### PEFT (Parameter-Efficient Fine-Tuning)
Updating only a small fraction of parameters.
- **LoRA (Low-Rank Adaptation)**: Inserting small, trainable matrices into the Transformer layers.
- **Adapter Layers**: Adding small modules between existing layers.

---

## 3. Best Practices 🧪

1.  **Lower Learning Rates**: Usually $2e-5$ to $5e-5$. Using a high rate can cause "Catastrophic Forgetting."
2.  **Learning Rate Decay**: Gradually decreasing the rate during training.
3.  **Warmup Steps**: Increasing the rate linearly at the start to stabilize the training.

---

## 🛠️ Essential Snippet (Hugging Face Trainer)

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
)

trainer.train()
```

---

## ⚖️ When to Stop?
Monitor **Validation Loss**. If it starts to increase while training loss decreases, your model is **overfitting** to your task-specific data.
