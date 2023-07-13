import torch.nn as nn
from transformers import BertModel


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear1 = nn.Linear(768, 2)
        # self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Softmax(dim=1)

    def forward(self, inputs):
        # with torch.no_grad():
        output = self.bert(input_ids=inputs['input_ids'].squeeze(1), attention_mask=inputs['attention_mask'].squeeze(1))
        # if not infer:
        #     output = self.dropout(output)
        # output = torch.squeeze(output[:, -1, :])
        output = self.linear1(output[1])
        output = self.out(output)
        return output
