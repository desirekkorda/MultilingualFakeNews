
import torch
import torch.nn as nn

from transformers import AutoModel

class XLMRClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            "xlm-roberta-base"
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            768,
            2
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        embedding = outputs.last_hidden_state.mean(
            dim=1
        )

        embedding = self.dropout(
            embedding
        )

        logits = self.fc(
            embedding
        )

        return logits
