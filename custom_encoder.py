import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, PreTrainedModel, BertConfig

# suggested revision for batch processing
class DualBertForStance(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config):
        super().__init__(config)
        self.post_encoder = BertModel(config)
        self.comment_encoder = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)

    def forward(
        self,
        post_input_ids=None,
        post_attention_mask=None,
        comment_input_ids=None,
        comment_attention_mask=None,
        labels=None,
    ):
        # Encode post
        post_outputs = self.post_encoder(
            input_ids=post_input_ids,
            attention_mask=post_attention_mask,
        )
        post_vec = post_outputs.pooler_output  # [batch, hidden]

        # Encode comment
        comment_outputs = self.comment_encoder(
            input_ids=comment_input_ids,
            attention_mask=comment_attention_mask,
        )
        comment_vec = comment_outputs.pooler_output  # [batch, hidden]

        combined = torch.cat([post_vec, comment_vec], dim=-1)
        logits = self.classifier(combined)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


class BertForStance3Way(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)  # encoder
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 3)  # 三分类

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
