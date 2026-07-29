import torch
import torch.nn as nn

from transformers import (
    XLMRobertaModel,
    XLMRobertaPreTrainedModel
)

from transformers.modeling_outputs import SequenceClassifierOutput


class XLMRMeanPoolingClassifier(
    XLMRobertaPreTrainedModel
):

    def __init__(self, config):

        super().__init__(config)

        self.num_labels = config.num_labels

        self.roberta = XLMRobertaModel(
            config,
            add_pooling_layer=False
        )

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            config.hidden_size,
            config.num_labels
        )

        self.post_init()

    def masked_mean_pooling(
        self,
        last_hidden_state,
        attention_mask
    ):

        mask = attention_mask.unsqueeze(-1).float()

        embeddings = last_hidden_state * mask

        summed = embeddings.sum(dim=1)

        counts = mask.sum(dim=1)

        pooled = summed / counts.clamp(min=1e-9)

        return pooled

    def forward(

        self,

        input_ids=None,

        attention_mask=None,

        labels=None,

    ):

        outputs = self.roberta(

            input_ids=input_ids,

            attention_mask=attention_mask

        )

        pooled = self.masked_mean_pooling(

            outputs.last_hidden_state,

            attention_mask

        )

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        loss = None

        if labels is not None:

            loss_fn = nn.CrossEntropyLoss()

            loss = loss_fn(

                logits,

                labels

            )

        return SequenceClassifierOutput(

            loss=loss,

            logits=logits

        )