from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import json

model_path="models/roberta-base-jd-binary"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = pipeline("sentiment-analysis", model=model_path)

print("loaded model")


with open("weibo2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def clean_text(text: str) -> str:
    return text.strip()

def get_sentiment(classifier,text):
    return classifier(text)

# Put all comments/replies into a list
sentiment_data=data
for c in sentiment_data["comments"]:
    res=get_sentiment(classifier,c["text"])[0]
    c["sent_label"]=res["label"]
    c["sent_score"]=res["score"]
    print(f"{c["text"]}\n{c['sent_label']}: {c['sent_score']}")
    if "replies" in c:
        for r in c["replies"]:
            res = get_sentiment(classifier,r["text"])[0]
            r["sent_label"] = res["label"]
            r["sent_score"] = res["score"]

# print(sentiment_data)
# print(*comments, sep="\n")
sent_out_path="weibo_example_sentiment.json"
with open(sent_out_path, "w", encoding="utf-8") as f:
    json.dump(sentiment_data, f, indent=2, ensure_ascii=False)

