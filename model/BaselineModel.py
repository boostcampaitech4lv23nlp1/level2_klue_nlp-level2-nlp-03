import torch
import torch.nn as nn
import einops as ein
from transformers import T5ForConditionalGeneration, AutoModel, AutoTokenizer

class BaselineModel(nn.Module):
    """_summary_
    베이스라인 모델입니다.
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
            self.model.resize_token_embeddings(AutoTokenizer.from_pretrained(self.model_name).vocab_size + add_token_num)
        self.regressor = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(self.model.config.hidden_size, self.num_labels)
        )

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooler = self.model(input_ids=input_ids, attention_mask=attention_mask).to_tuple()
        logits = self.regressor(pooler)
        
        return logits
    
class HiddenModel(nn.Module):
    """_summary_
    last_hidden_state를 사용하는 모델입니다.
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
    
class T5Model(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask, tgt_input_ids:None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels = tgt_input_ids)
        return outputs
    
class PLMRNNModel(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate, add_token_num=None):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)

        if add_token_num:
            self.model.resize_token_embeddings(AutoTokenizer.from_pretrained(model_name).vocab_size + add_token_num)

        self.lstm = nn.LSTM(input_size=self.model.config.hidden_size,
                            hidden_size=self.model.config.hidden_size,
                            num_layers=3,
                            bidirectional=True,
                            batch_first=True)
        
        self.gru = nn.GRU(input_size=self.model.config.hidden_size,
                          hidden_size=self.model.config.hidden_size,
                          num_layers=3,
                          batch_first=True,
                          bidirectional=True)
        
        self.activation = nn.Tanh()
        self.regressor = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.model.config.hidden_size*2, self.num_labels))
        
    def forward(self, input_ids, attention_mask):
        if 't5' in self.model_name:
            outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
            _, hidden = self.gru(outputs) 
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
            # _, (hidden, _) = self.lstm(outputs) 
            _, hidden = self.gru(outputs) 
        outputs = torch.cat([hidden[-1], hidden[-2]], dim=1)
        outputs= self.activation(outputs)
        logits = self.regressor(outputs)

        return logits
    
