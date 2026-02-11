# src/training/eval_classifier.py
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=-1)
    test = load_dataset("parquet", data_files="data_processed/test.parquet")["train"]
    texts = test["text"]
    labels = test["mood"]
    preds=[]
    for t in texts:
        out = pipe(t, top_k=1)
        preds.append(out[0]["label"])
    print(classification_report(labels, preds))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="models/mood_intent_classifier/mood")
    args = parser.parse_args()
    main(args)
