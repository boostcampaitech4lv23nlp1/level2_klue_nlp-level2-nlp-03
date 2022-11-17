import gc
import torch
import numpy as np
import wandb
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm.auto import tqdm


def train(model, loss_fn, optimizer, scheduler, train_dataloader, valid_dataloader, EPOCHS, device):
    min_val_loss = 1
    for epoch in range(EPOCHS):
        gc.collect()
        model.train() 
        epoch_loss = 0
        steps = 0
        pbar = tqdm(train_dataloader)
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            steps += 1
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["labels"].to(device, dtype=torch.int64).squeeze(-1)
            logits = model(input_ids, attention_mask)

            prob = F.softmax(logits, dim=-1)
            # loss = loss_fn(logits.detach().cpu(), label.detach().cpu())
            loss = F.cross_entropy(logits, label)
            loss.backward()
            epoch_loss += loss.detach().cpu().numpy().item()
            
            optimizer.step()
            # scheduler.step()
            
            pbar.set_postfix({'loss': epoch_loss/steps, 
                        "lr": optimizer.param_groups[0]["lr"]
                    })

            wandb.log({
                'epoch' : epoch
                ,'train_loss':epoch_loss/steps})

        pbar.close()
        val_loss = 0
        val_steps = 0
        total_val_score=0
        
        with torch.no_grad():
            model.eval()
            for valid_batch in tqdm(valid_dataloader):
                input_ids = valid_batch["input_ids"].to(device)
                attention_mask = valid_batch["attention_mask"].to(device)
                label = valid_batch["labels"].to(device, dtype=torch.int64).squeeze(-1)
                
                logits = model(input_ids, attention_mask)

                prob = F.softmax(logits, dim=-1).detach().cpu().numpy()

                val_steps += 1
                
                loss = F.cross_entropy(logits, label)
                val_loss += loss.detach().cpu().numpy().item()                
                
                preds = np.argmax(prob, axis=-1)
                label = label.squeeze().detach().cpu().numpy()
                total_val_score += f1_score(label, preds, average='micro').item()
            
            val_loss /= val_steps
            total_val_score /= val_steps
            print(f"Epoch [{epoch+1}/{EPOCHS}] Score : {total_val_score}")
            print(f"Epoch [{epoch+1}/{EPOCHS}] Val_loss : {val_loss}")

            if min_val_loss > val_loss:
                print('save checkpoint!')
                torch.save(model.state_dict(), 'model.pt')
                min_val_loss = val_loss

            wandb.log({
                'epoch' : epoch
                ,'val_loss':val_loss})
            wandb.log({
                'epoch' : epoch 
                ,'val_acc':total_val_score})

    del train_dataloader, valid_dataloader
    gc.collect()