import torch
from transformers import AutoTokenizer
import os

from models.multitask_transformer import TransformerMultiTaskModel

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define model parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerMultiTaskModel("bert-base-uncased", num_sent_classes=3, num_ner_tags=9).to(device)

# Load saved model
model_load_path = "models/multitask_transformer.pth"
if os.path.exists(model_load_path):
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    print(f"Model loaded from {model_load_path}")
else:
    raise FileNotFoundError(f"Model file not found at {model_load_path}")

model.eval()

# 🔹 Run inference on a sample sentence
def predict(sentence):
    encoded = tokenizer(sentence, truncation=True, padding="max_length", max_length=64, return_tensors="pt")

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        sent_logits, ner_logits = model(input_ids, attention_mask)

        # Sentence classification prediction
        sent_pred = torch.argmax(sent_logits, dim=1).item()

        # Named Entity Recognition (NER) prediction
        ner_preds = torch.argmax(ner_logits, dim=2).cpu().numpy()

    return sent_pred, ner_preds

# Example usage
sentence = "Can you help me?"
sent_label, ner_predictions = predict(sentence)

print(f"🔹 Sentence: {sentence}")
print(f"🔹 Predicted Sentence Class: {sent_label}")
print(f"🔹 Predicted NER Labels: {ner_predictions}")
