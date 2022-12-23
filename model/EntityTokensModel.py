import torch
import torch.nn as nn
import einops as ein
from transformers import AutoModel, AutoTokenizer
    
class EntityTokensModel(nn.Module):
    """_summary_
    last_hidden_state 중에서 첫번째 entity 토큰만을 사용하는 모델입니다.
    숫자 인덱스로 접근해 두 히든 스테이트를 뽑아서 분류해줍니다.
    """
    def __init__(self, model_name, num_labels, dropout_rate, add_token_num=None):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        
        if add_token_num:
            self.model.resize_token_embeddings(AutoTokenizer.from_pretrained(model_name).vocab_size + add_token_num)
            
        self.regressor = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.hidden_size, self.num_labels)
        )
        
    def forward(self, input_ids, attention_mask, entity_embed1, entity_embed2):
        last_hidden_state = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        idx = torch.arange(input_ids.size(0)).to(input_ids.device)
        subj_emb = last_hidden_state[idx, entity_embed1]
        obj_emb = last_hidden_state[idx, entity_embed2]
        output = torch.cat((subj_emb, obj_emb), dim=-1)
        logits = self.regressor(output)
        
        return logits