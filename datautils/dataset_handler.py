import pandas as pd
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# this is to load sentence classification dataset from the csv file
def load_sentence_classification_data(file_path):
    df = pd.read_csv(file_path)
    sentences = df["doc"].tolist()
    sent_labels = df["target"].tolist()  # Target is the integer label for class
    return sentences, sent_labels



# this is to load the NER dataset from the conll2003 dataset
def load_ner_data():
    dataset = load_dataset("conll2003")

    sentences = []
    ner_labels = []

    for data in dataset['train']:
        words = data['tokens']
        labels = data['ner_tags']

        sentences.append(" ".join(words))  # Convert word list to sentence
        ner_labels.append(labels)

    return sentences, ner_labels


class MultiTaskDataset(Dataset):
    def __init__(self, sentences, sent_labels, ner_labels, tokenizer, max_length=64):
        self.sentences = sentences
        self.sent_labels = sent_labels
        self.ner_labels = ner_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        sent_label = self.sent_labels[idx] if self.sent_labels else None
        ner_label = self.ner_labels[idx] if self.ner_labels else None

        encoded = self.tokenizer(sentence, truncation=True, padding="max_length",
                                 max_length=self.max_length, return_tensors="pt")

        item = {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }

        if sent_label is not None:
            item['sent_label'] = torch.tensor(sent_label, dtype=torch.long)

        if ner_label is not None:
            padded_ner_labels = ner_label + [-1] * (self.max_length - len(ner_label))  # Pad to max length
            item['ner_labels'] = torch.tensor(padded_ner_labels, dtype=torch.long)

        return item






