import torch
import torch.nn as nn
import einops as ein
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration
    
class LastHiddenGRUModel(nn.Module):
    """_summary_
    last_hidden_state 중에서 첫번째 entity 토큰만을 사용하는 모델입니다.
    숫자 인덱스로 접근해 두 히든 스테이트를 뽑아서 분류해줍니다.
    """
    def __init__(self, model_name, num_labels, dropout_rate, add_token_num=None):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.model_name = model_name
        
        if 't5' in self.model_name:
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).get_encoder()
        else:
            self.model = AutoModel.from_pretrained(self.model_name)

        if add_token_num:
            self.model.resize_token_embeddings(AutoTokenizer.from_pretrained(model_name).vocab_size + add_token_num)
            
        self.gru = nn.GRU(input_size=self.model.config.hidden_size,
                          hidden_size=self.model.config.hidden_size,
                          num_layers=3,
                          batch_first=True,
                          bidirectional=True)

        self.regressor = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.model.config.hidden_size * 2, self.num_labels)
        )

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        _, hidden = self.gru(last_hidden_state)
        output = torch.cat([hidden[-1], hidden[-2]], dim=1)
        logits = self.regressor(output)
        
        return logits