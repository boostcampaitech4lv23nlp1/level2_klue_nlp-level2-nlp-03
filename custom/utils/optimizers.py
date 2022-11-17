from transformers import AdamW, get_linear_schedule_with_warmup


def fetch_scheduler(model, train_dataloader, LEARNING_RATE, WEIGHT_DECAY, EPOCHS):
    optimizer = AdamW(model.parameters(),
                        lr=LEARNING_RATE)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=EPOCHS*(len(train_dataloader)))
    
    return optimizer, scheduler