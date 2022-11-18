import gc
import pandas as pd
import wandb
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_loader.custom_dataloader import Customdataset
from models.encoder_model import EncoderModel
import metrics.metrics as Metrics
from utils.optimizers import fetch_scheduler
from utils.loss import CELoss
from trainer import train as trainer


device='cuda'
MAX_LEN = 160

def get_train_dataloader(tokenizer, train_data, valid_data, BATCH_SIZE):
    train_dataset = Customdataset(train_data, tokenizer,True)
    val_dataset = Customdataset(valid_data, tokenizer,True)
    num_to_label = train_dataset.num_to_label
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)
    valid_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False)
    
    return train_dataloader, valid_dataloader, num_to_label 

# load data
train_path = '/opt/ml/custom/train/train.csv'
df_data = pd.read_csv(train_path)

train, dev = train_test_split(df_data, test_size=0.2,shuffle=True)
train.reset_index(drop=True, inplace=True)
dev.reset_index(drop=True, inplace=True)

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# wandb set
group_name='bert'
experiment_name='baseline'

config = {'EPOCHS':10,
          'LEARNING_RATE':5e-5,
          'BATCH_SIZE':64,
          'WEIGHT_DECAY':1e-5,
          'drop_rate':0.2,
          'MODEL_NAME':"klue/bert-base"}

wandb.init(project='test-project',
        config=config, group=group_name,
        name=experiment_name,
        entity="nlp6")

## metrics 이름을 metric_list에 넣어놓고 metrics 딕셔너리에 이름과 함수를 초기화
metric_list = ['klue_re_micro_f1', 'klue_re_auprc']
metrics = {metric_list : getattr(Metrics, metric) for metric in metric_list}

loss_fn = torch.nn.CrossEntropyLoss()
# loss_fn = CELoss
tokenizer = AutoTokenizer.from_pretrained(wandb.config.MODEL_NAME)
model = EncoderModel(wandb.config.MODEL_NAME, num_labels=30,drop_rate=wandb.config.drop_rate).to(device)
train_dataloader, valid_dataloader, num_to_label = get_train_dataloader(tokenizer, train, dev, wandb.config.BATCH_SIZE)
optimizer, scheduler= fetch_scheduler(model, train_dataloader, wandb.config.LEARNING_RATE, wandb.config.WEIGHT_DECAY, wandb.config.EPOCHS)

print("Training")
print("="*100)

trainer(model, 
    loss_fn,
    metrics,
    optimizer, 
    scheduler,
    train_dataloader, 
    valid_dataloader,
    wandb.config.EPOCHS,
    device)
    
del model, train_dataloader, valid_dataloader
_ = gc.collect()
torch.cuda.empty_cache()