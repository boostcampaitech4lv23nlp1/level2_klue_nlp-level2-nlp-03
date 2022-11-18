import torch
from torch import nn
from transformers import AutoModel

class Baseline(nn.Module):
    def __init__(self, model_name, num_labels, drop_rate):
        super(EncoderModel, self).__init__()
        self.drop_rate=drop_rate
        self.model = AutoModel.from_pretrained(
                        pretrained_model_name_or_path=model_name)
        self.regressor = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(self.model.config.hidden_size, num_labels))
        
        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #                 pretrained_model_name_or_path=model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)[1]
        logits = self.regressor(outputs)
        # logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits

class EncoderModel(nn.Module):
    def __init__(self, model_name, num_labels, drop_rate):
        super(EncoderModel, self).__init__()
        self.drop_rate=drop_rate
        self.model = AutoModel.from_pretrained(
                        pretrained_model_name_or_path=model_name)
        
        self.biGRU = nn.GRU(input_size=self.model.config.hidden_size,
                            hidden_size=self.model.config.hidden_size,
                        num_layers=3,
                        dropout=drop_rate, 
                        batch_first=True,
                        bidirectional=True)

        self.activation = nn.Tanh()

        self.regressor = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(self.model.config.hidden_size*2, num_labels))

        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #                 pretrained_model_name_or_path=model_name, num_labels=num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        #### Custom
        _, hc = self.biGRU(outputs) 
        outputs = torch.cat([hc[-1], hc[-2]], dim=1)
        outputs= self.activation(outputs)
        ####
        logits = self.regressor(outputs)
        # logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits