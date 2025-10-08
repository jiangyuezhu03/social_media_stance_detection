from transformers import BertTokenizer, BertForSequenceClassification
import torch
from data_processing import get_comments_and_replies,get_post
from custom_encoder import DualBertForStance, BertForStance3Way

model_path = "models/stance_ch"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

def predict_stance(post, comment, tokenizer, model):
    # text = post + " [SEP] " + comment
    # inputs = tokenizer(
    #     text,
    #     return_tensors="pt",
    #     truncation=True,
    #     padding=True,
    #     max_length=512)
    inputs = tokenizer(
        post,
        comment,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()

    labels = {"0": "OPPOSING", "1": "SUPPORTIVE"}
    confidence = predictions[0][predicted_class].item()

    return {
        "stance": labels[str(predicted_class)],
        "confidence": confidence
    }

def print_prediction(post, comment_list,tokenizer, model):
    print(f"Post: {post}")
    for comment in comment_list:
        result = predict_stance(post, comment,tokenizer, model)

        print(f"Comment: {comment}")
        print(f"Stance: {result['stance']}, Confidence: {result['confidence']:.4f}")

post = "姐妹们，这家店的服务真的太好了，而且还不贵，就在地铁站旁边。"
comments = [
    "又是广告",
    "这家店我去过，感觉一般",
    "体验也很好"
]
print_prediction(post,comments,tokenizer, model)

