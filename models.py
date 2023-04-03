import torch
import torch.nn as nn
from transformers import BartConfig, BartModel

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        config = BartConfig.from_pretrained("/root/autodl-tmp/pretrain/")
        self.model = BartModel(config)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(config.d_model, 2)
        self.out = nn.Softmax(dim=0)

    def forward(self, inputs, attn_mask, infer=False):
        output = self.model(input_ids=inputs, attention_mask=attn_mask).last_hidden_state
        if not infer:
            output = self.dropout(output)
        output = self.linear(output)
        output = torch.squeeze(output[:, -1, :])
        output = self.out(output)
        return output
