import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import (
    BertModel,
    BertConfig,
)
from .model_helper import NERModelOutput


class NERModel(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(NERModel, self).__init__()
        self.num_labels = num_labels

        # get bert model
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.config = BertConfig.from_pretrained(bert_model_name)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        loss_fn = CrossEntropyLoss()
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fn(active_logits, active_labels)
            else:
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return NERModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )