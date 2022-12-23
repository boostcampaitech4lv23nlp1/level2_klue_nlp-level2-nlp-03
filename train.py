import torch
import argparse
import pandas as pd
import pickle
from omegaconf import OmegaConf

from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import StratifiedKFold

import dataloader as DataModule
import trainer as Trainer
import model as Model

import torch.optim as optim
import utils.loss as Criterion
import utils.metric as Metric
import wandb
from utils.seed_setting import seed_setting

config = None

def main():
    print(config)
    seed_setting(config.train.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)  
    data = pd.read_csv(config.data.train_path)
    with open('data/dict_label_to_num.pkl', 'rb') as f:
        dict_label_to_num = pickle.load(f)
    data['label'] = data['label'].apply(lambda x: dict_label_to_num[x])
    skf = StratifiedKFold(n_splits=config.data.n_splits) # train : valid = 0.8 : 0.2
    train_index, val_index = next(iter(skf.split(data, data['label'])))
    
    # 데이터셋 로드 클래스를 불러옵니다.
    # marker 설정은 ['entity_mask', 'entity_marker', 'typed_entity_marker', 'entity_marker_punct', typed_entity_marker_punct]
    print('='*50, f'현재 적용되고 있는 데이터 클래스는 {config.model.data_class}입니다.', '='*50, sep='\n\n')
    train = getattr(DataModule, config.model.data_class)(
        mode = "train",
        data=data.iloc[train_index,:],
        tokenizer=tokenizer,
        max_length=config.train.max_length,
        entity_marker_mode= config.data.get('entity_marker_mode'))
    valid = getattr(DataModule, config.model.data_class)(
        mode = "train",
        data=data.iloc[val_index,:],
        tokenizer=tokenizer,
        entity_marker_mode= config.data.get('entity_marker_mode'),
        max_length=config.train.max_length)

    train_dataloader = DataLoader(train, batch_size= config.train.batch_size, pin_memory=True, shuffle=True)
    valid_dataloader = DataLoader(valid, batch_size= config.train.batch_size, pin_memory=True, shuffle=False)

    assert torch.cuda.is_available(), "GPU를 사용할 수 없습니다."
    device = torch.device('cuda')
    
    # 모델 아키텍처를 불러옵니다.
    print('='*50,f'현재 적용되고 있는 모델 클래스는 {config.model.model_class}입니다.', '='*50, sep='\n\n')
    model = getattr(Model, config.model.model_class)(
        model_name = config.model.model_name,
        num_labels=30,
        dropout_rate = config.model.dropout_rate,
        add_token_num = config.data.get('entity_marker_num')
        ).to(device)
    
    criterion = getattr(Criterion, config.model.loss)
    metric_list = {config.model.metric_list[f'metric{i+1}'] : getattr(Metric, config.model.metric_list[f'metric{i+1}']) for i in range(len(config.model.metric_list))}
    optimizer = getattr(optim, config.model.optimizer)(model.parameters(), lr=config.train.learning_rate)
    
    lr_scheduler = None
    epochs = config.train.max_epoch
    
    print('='*50,f'현재 적용되고 있는 트레이너는 {config.model.trainer}입니다.', '='*50, sep='\n\n')
    trainer = getattr(Trainer, config.model.trainer)(
            model = model,
            criterion = criterion,
            metric = metric_list,
            optimizer = optimizer,
            device = device,
            save_dir = config.model.saved_dir,
            train_dataloader = train_dataloader,
            valid_dataloader = valid_dataloader,
            lr_scheduler=lr_scheduler,
            epochs=epochs,
            tokenizer = tokenizer
        )
    
    trainer.train()

def wandb_sweep():
    with wandb.init() as run:
        # update any values not set by sweep
        # run.config.setdefaults(config)
        for k, v in run.config.items():
            OmegaConf.update(config, k, v)
        main()

if __name__=='__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='roberta_large/roberta_large_typed_entity_marker_punct_tokens')
    parser.add_argument('--wandb', type=str, default='init')
    args, _ = parser.parse_known_args()
    ## ex) python3 train.py --config baseline
    
    config = OmegaConf.load(f'./configs/{args.config}.yaml')
    
    # wandb 설정을 해주지 않으면 오류가 납니다
    wandb_config = OmegaConf.load(f'./configs/train/wandb_{args.wandb}.yaml')
    
    print(f'사용할 수 있는 GPU는 {torch.cuda.device_count()}개 입니다.')
    
    wandb.login()
    
    if wandb_config.get('sweep'):
        sweep_config = OmegaConf.to_object(wandb_config.sweep)
        sweep_id = wandb.sweep(
                sweep=sweep_config,
                entity=wandb_config.entity,
                project=wandb_config.project)
        wandb.agent(sweep_id=sweep_id, function=wandb_sweep, count=wandb_config.count)
        
    else:
        wandb.init(
                entity=wandb_config.entity,
                project=wandb_config.project,
                group=wandb_config.group,
                name=wandb_config.experiment)
        # wandb.config = config
        main()
    