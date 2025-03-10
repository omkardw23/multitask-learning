import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer
import torch.nn as nn

from datautils.dataset_handler import *
from models.multitask_transformer import *
from tqdm import tqdm

print("CUDA Available:", torch.cuda.is_available())
print("Current Device:", torch.cuda.current_device())
print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))


# Load datasets
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sentences_cls, sent_labels = load_sentence_classification_data("data/questions_vs_statements_v1.0.csv")
sentences_ner, ner_labels = load_ner_data()

# Create datasets & dataloaders
cls_dataset = MultiTaskDataset(sentences_cls, sent_labels, None, tokenizer)
ner_dataset = MultiTaskDataset(sentences_ner, None, ner_labels, tokenizer)

cls_loader = DataLoader(cls_dataset, batch_size=8, shuffle=True, num_workers=0,pin_memory=True)
ner_loader = DataLoader(ner_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerMultiTaskModel("bert-base-uncased", num_sent_classes=3, num_ner_tags=9).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion_sent = nn.CrossEntropyLoss()
criterion_ner = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding

print("Running model warm-up test on GPU...")
dummy_input = torch.randint(0, 100, (1, 64)).to("cuda")
dummy_mask = torch.ones((1, 64)).to("cuda")
model(dummy_input, dummy_mask)  # First CUDA call
print("GPU Ready! Training will now start.")


# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0

    print(f"\nEpoch {epoch+1}/{epochs}")  # Print current epoch

    # Track progress for sentence classification training
    cls_progress_bar = tqdm(cls_loader, desc=f"🔹 Training Sentence Classification | Epoch {epoch+1}")
    # Train sentence classification
    for batch in cls_loader:
        print(f"Processing batch {batch}/{len(cls_loader)} for Sentence Classification")
        optimizer.zero_grad()
        input_ids, attention_mask, sent_labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['sent_label'].to(device)

        sent_logits, _ = model(input_ids, attention_mask)
        loss_sent = criterion_sent(sent_logits, sent_labels)
        loss_sent.backward()
        optimizer.step()
        total_loss += loss_sent.item()

        cls_progress_bar.set_postfix({"Loss": loss_sent.item()})  # Update progress bar

    ner_progress_bar = tqdm(ner_loader, desc=f"Training NER | Epoch {epoch+1}")
    # Train NER
    for batch in ner_loader:
        print(f"Processing batch {batch}/{len(ner_loader)} for NER")
        optimizer.zero_grad()
        input_ids, attention_mask, ner_labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['ner_labels'].to(device)

        _, ner_logits = model(input_ids, attention_mask)
        loss_ner = criterion_ner(ner_logits.view(-1, 9), ner_labels.view(-1))  # Flatten for loss calculation
        loss_ner.backward()
        optimizer.step()
        total_loss += loss_ner.item()

        ner_progress_bar.set_postfix({"Loss": loss_ner.item()})  # Update progress bar

    print(f"Epoch {epoch+1}/{epochs} completed. Total Loss: {total_loss:.4f}\n")




model.eval()
sentence = ["Can you help me?"]
encoded = tokenizer(sentence, truncation=True, padding="max_length", max_length=64, return_tensors="pt")

input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

with torch.no_grad():
    sent_logits, ner_logits = model(input_ids, attention_mask)

    sent_pred = torch.argmax(sent_logits, dim=1).item()
    ner_preds = torch.argmax(ner_logits, dim=2).cpu().numpy()

print("Sentence Type Prediction:", sent_pred)
print("NER Predictions:", ner_preds)