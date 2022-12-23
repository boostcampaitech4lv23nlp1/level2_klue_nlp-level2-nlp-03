import torch
import torch.nn as nn
import einops as ein
from transformers import AutoModel, AutoTokenizer

class LastHiddenModel(nn.Module):
    """_summary_
    last_hidden_state를 사용하는 모델입니다.
    last_hidden_state를 사용하는 모델은 Bert, Roberta, Electra, T5 등등입니다.
    """
    def __init__(self, model_name, num_labels, dropout_rate, add_token_num=None):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        
        self.model = AutoModel.from_pretrained(model_name)
        if add_token_num:
            self.model.resize_token_embeddings(AutoTokenizer.from_pretrained(model_name).vocab_size + add_token_num)
        self.regressor = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.model.config.hidden_size, self.num_labels)
        )
        
    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooler = self.model(input_ids=input_ids, attention_mask=attention_mask).to_tuple()

        logits = self.regressor(pooler)
        
        return logits