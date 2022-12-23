import torch
import torch.nn as nn
import einops as ein
from transformers import AutoModel, AutoTokenizer

class EntityMaskModel(nn.Module):
    """_summary_
    last hidden state와 entity mask를 사용하는 모델입니다.
    """
    def __init__(self, model_name, num_labels, dropout_rate, add_token_num=None):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        
        self.model = AutoModel.from_pretrained(model_name)
        
        if add_token_num:
            self.model.resize_token_embeddings(AutoTokenizer.from_pretrained(model_name).vocab_size + add_token_num)
            
        self.regressor = nn.Sequential(
            nn.Linear(2 * self.model.config.hidden_size, self.model.config.hidden_size), # [batch, 1, hidden_size] -> [batch, hidden_size]
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.model.config.hidden_size, self.num_labels)
        )
        
    def forward(self, input_ids, attention_mask, entity_mask1, entity_mask2):
        last_hidden_state, pooler = self.model(input_ids=input_ids, attention_mask=attention_mask).to_tuple()
        
        entity_layer1 = self.entity_mask_layer(entity_mask1, last_hidden_state) # [batch, hidden_size]
        entity_layer2 = self.entity_mask_layer(entity_mask2, last_hidden_state) # [batch, hidden_size]
        
        h = torch.cat((entity_layer1, entity_layer2), dim=-1) #[batch, 2, hidden_size]

        logits = self.regressor(pooler)
        
        return logits
    
    def entity_mask_layer(self, entity_mask, last_hidden_state): 
        e_mask_unsqueeze = ein.rearrange(entity_mask, 'batch seq_len -> batch () seq_len') # (batch, 1, sequecen_length)
        length_tensor = ein.rearrange((entity_mask != 0).sum(dim=1), 'batch -> batch ()')
        
        sum_vector = ein.rearrange(torch.bmm(e_mask_unsqueeze.float(), last_hidden_state), 'batch 1 hidden -> batch hidden')
        avg_vector = sum_vector.float() / length_tensor.float()
        
        return avg_vector