import gc
import pandas as pd
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_loader.custom_dataloader import Customdataset
from models.encoder_model import EncoderModel
from utils.optimizers import fetch_scheduler
from utils.loss import CELoss
from trainer import train as trainer


device='cuda'
MAX_LEN = 160

def get_test_dataloader(tokenizer, test_data, BATCH_SIZE):
    test_dataset = Customdataset(test_data, tokenizer,True)
    num_to_label = test_data.num_to_label
    
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False)
    
    return test_dataloader, num_to_label 

def func_num_to_label(pred_list, num_to_label):
    labels = []
    for x in pred_list:
        labels.append(num_to_label[x])
    return labels

def inference(model, test_dataloader):
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)

            probs = F.softmax(logits, dim=-1)
            preds = np.argmax(probs, axis=-1)

            all_preds.append(preds)
            all_probs.append(probs)
    return np.concatenate(all_preds).tolist(), np.concatenate(all_probs, axis=0).tolist()

# load data
test_path = '/opt/ml/custom/test/test_data.csv'
df_data = pd.read_csv(test_path)
df_data.reset_index(drop=True, inplace=True)

tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

MODEL_NAME = 'klue/bert-base'

checkpoint = torch.load('model.pt')
model = EncoderModel(MODEL_NAME, num_labels=30,drop_rate=0.2).to(device)
model.load_state_dict(checkpoint)
test_dataset, num_to_label = get_test_dataloader(tokenizer, df_data, 1)

print("Inference")
print("="*100)

predictions, probablity = inference(model, test_dataset)
predictions = func_num_to_label(predictions, num_to_label)

output = df_data.assign(pred_label = predictions, probs = probablity)
output.to_csv('./prediction/submission.csv', index=False)