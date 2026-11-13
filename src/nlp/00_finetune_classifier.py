"""
Fine-tunes mDeBERTa-v3-base to flag first-person fraud complaints.

Settings follow Section 3.3 of the paper: stratified 80/20 split, four epochs, batch size 16, weight decay 0.01,
learning rate 2e-5. The annotated training posts contain text and are not distributed; this script expects them at
data/private/annotated_posts.csv with the columns  text, label  (1 = first-person fraud complaint, 0 = otherwise).
"""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

MODEL = "microsoft/mdeberta-v3-base"
SEED = 42


class Posts(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        return item


def metrics(p):
    pred = p.predictions.argmax(-1)
    pr, rc, f1, _ = precision_recall_fscore_support(p.label_ids, pred, average="binary")
    return {"precision": pr, "recall": rc, "f1": f1}


if __name__ == "__main__":
    df = pd.read_csv("../../data/private/annotated_posts.csv").dropna(subset=["text"])
    tr, va = train_test_split(df, test_size=0.2, stratify=df.label, random_state=SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = lambda d: tok(d.text.astype(str).tolist(), truncation=True, padding=True, max_length=128)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    args = TrainingArguments(output_dir="../../data/private/classifier", num_train_epochs=4, per_device_train_batch_size=16,
                             learning_rate=2e-5, weight_decay=0.01, eval_strategy="epoch", save_strategy="no", seed=SEED)
    trainer = Trainer(model=model, args=args, train_dataset=Posts(enc(tr), tr.label.values),
                      eval_dataset=Posts(enc(va), va.label.values), compute_metrics=metrics)
    trainer.train()
    print(trainer.evaluate())
    trainer.save_model("../../data/private/classifier/final")
