from torch import nn
from transformers import AutoModel

import config


class ReviewAnalyzeModel(nn.Module):
    def __init__(self, is_freeze_bert=True):
        super().__init__()

        self.bert = AutoModel.from_pretrained(config.PRETRAINED_MODEL_NAME)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)

        if is_freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # input_ids: [batch_size, seq_len] attention_mask: [batch_size, seq_len]

        output = self.bert(input_ids, attention_mask)
        # output is dict with keys: last_hidden_state, pooler_output, etc.
        last_hidden_state = output.last_hidden_state
        # last_hidden_state.shape is [batch_size, seq_len, hidden_size]
        cls_token = last_hidden_state[:, 0, :]
        # cls_token.shape is [batch_size, hidden_size]
        output = self.linear(cls_token)
        # output.shape is [batch_size, 1]
        output = output.squeeze(-1)
        # output.shape is [batch_size]
        return output


if __name__ == '__main__':
    model = ReviewAnalyzeModel()
    print(model)
