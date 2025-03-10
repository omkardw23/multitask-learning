import torch.nn as nn
from transformers import AutoModel

class TransformerMultiTaskModel(nn.Module):
    def __init__(self, transformer_name, num_sent_classes, num_ner_tags):
        super(TransformerMultiTaskModel, self).__init__()

        # Load Transformer model (e.g., BERT)
        self.transformer = AutoModel.from_pretrained(transformer_name)
        hidden_size = self.transformer.config.hidden_size  # Typically 768

        # Sentence Classification Head (CLS token)
        self.sent_classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_sent_classes)
        )

        # Named Entity Recognition Head (token classification)
        self.ner_classifier = nn.Linear(hidden_size, num_ner_tags)

    def forward(self, input_ids, attention_mask):
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Sentence classification (using [CLS] token)
        cls_embedding = hidden_states[:, 0, :]
        sent_logits = self.sent_classifier(cls_embedding)  # (batch, num_sent_classes)

        # Named Entity Recognition (per-token classification)
        ner_logits = self.ner_classifier(hidden_states)  # (batch, seq_len, num_ner_tags)

        return sent_logits, ner_logits
