# src/training/train_classifier.py
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
from pathlib import Path

def compute_metrics(eval_pred):
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1_macro": metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    }

def main(args):
    model_name = args.model_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train = load_dataset("parquet", data_files="data_processed/train.parquet")["train"]
    test = load_dataset("parquet", data_files="data_processed/test.parquet")["train"]
    label_list = sorted(list(set(train["mood"])))
    label2id = {l:i for i,l in enumerate(label_list)}
    id2label = {i:l for l,i in label2id.items()}

    def preprocess(examples):
        toks = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        toks["labels"]=[label2id.get(x, 0) for x in examples["mood"]]
        return toks

    train = train.map(preprocess, batched=True, remove_columns=train.column_names)
    test = test.map(preprocess, batched=True, remove_columns=test.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print("Saved model to", args.output_dir)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", default="bert-base-uncased")
    parser.add_argument("--output_dir", default="models/mood_intent_classifier/mood")
    args = parser.parse_args()
    main(args)
