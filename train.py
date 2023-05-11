import os
import fire
import warnings
import numpy as np
import torch
import bitsandbytes as bnb
from model import GPT, GPTConfig
from datasets import load_dataset
from transformers import AutoTokenizer #, AutoConfig, DataCollatorForLanguageModeling
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb
from accelerate import Accelerator

def get_tokenize_fn(tokenizer, seq_len):
    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            padding=False,
            truncation=False,
            return_special_tokens_mask=False,
        ).input_ids
        result = []
        for seq in tokens:
            result.extend(seq)
            result.append(tokenizer.eos_token_id)
        result = result[:(seq_len + 1) * (len(result) // (seq_len + 1))]
        seqs = np.array(result).reshape(-1, seq_len + 1)
        return {
            "input_ids": seqs[:, :-1],
            "targets": seqs[:, 1:]
        }
    return tokenize_function

def train(
    config_path="config/1b-default.yaml",
    tokenizer_path="EleutherAI/pythia-1.4b",
    train_dataset="andersonbcdefg/minipile_train_tokenized",
    val_dataset="andersonbcdefg/minipile_val_tokenized",
    pre_tokenized=True,
    max_batch_size = 2048,
    micro_batch_size = 32,
    initial_batch_size = 256,
    steps_per_batch_size = 16,
    max_lr = 3.0e-4,
    pct_start = 0.1,
    initial_div_factor = 5,
    final_div_factor = 25,
    num_epochs = 1,
    anneal_strategy = "linear", # "cosine" is the other option
    optim_8bit = False,
    precision="bf16",
    grad_clip = None,
    save_every = 10000,
    ckpt_dir = "/storage"
):
    # make it so that we specify either my GPT or gpt_neox in the config and load accordingly
    config = GPTConfig.from_yaml(config_path)
    model = GPT(config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset("andersonbcdefg/minipile_train_tokenized", split="train")
    train_dataset.set_format("pt")
    if val_dataset is not None:
        val_dataset = load_dataset("andersonbcdefg/minipile_val_tokenized", split="validation")
        val_dataset.set_format("pt")
    
    if not pre_tokenized:
        train_dataset = train_dataset.map(get_tokenize_fn(tokenizer, config.seq_len), batched=True, remove_columns=train_dataset.column_names)
        if val_dataset is not None:
            val_dataset = val_dataset.map(get_tokenize_fn(tokenizer, config.seq_len), batched=True, remove_columns=val_dataset.column_names)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True, num_workers=4, pin_memory=True) #, collate_fn=collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False) #, collate_fn=collate_fn)

    # Model, optimizer, scheduler
    model.to(torch.device("cuda"))
    if optim_8bit:
        if config.stable_embedding == False:
            warnings.warn("You are using 8-bit AdamW but stable embedding is not enabled. This is not recommended.")
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=max_lr, betas=(0.9, 0.95)) #, eps=1.0e-3) -- i really don't think this works
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), fused=True)
    scheduler_kwargs = {
        "max_lr": max_lr,
        "total_steps": (len(train_dataloader) + 10) * num_epochs,
        "pct_start": pct_start,
        "div_factor": initial_div_factor,
        "final_div_factor": final_div_factor,
        "anneal_strategy": anneal_strategy,
        "three_phase": False
    }
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_kwargs)

    wandb.init(
        project="train_gpt_1.4b",
        config={
            "model": config.__dict__,
            "max_lr": max_lr,
            "scheduler": scheduler_kwargs,
            "optim_8bit": optim_8bit,
            "micro_batch_size": micro_batch_size,
            "num_epochs": num_epochs,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "precision": precision
        }
    )

    accelerator = Accelerator(mixed_precision=precision)
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    # Training loop
    gradient_accumulation_steps = initial_batch_size // micro_batch_size
    micro_batches = 0
    optimizer_steps = 0
    model.train()
    
    for index, batch in enumerate(train_dataloader):
        input_ids, labels = batch["input_ids"], batch["targets"]
        if micro_batches < gradient_accumulation_steps - 1:
            with accelerator.no_sync(model):
                micro_batch_loss = model(input_ids, labels=labels)
                if isinstance(micro_batch_loss, CausalLMOutputWithPast):
                    micro_batch_loss = micro_batch_loss.loss
                accelerator.backward(micro_batch_loss)
                if grad_clip is not None:
                    accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            wandb.log({
                "micro_batch_loss": micro_batch_loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": gradient_accumulation_steps * micro_batch_size,
                "tokens": index * 512 * micro_batch_size
            })
            micro_batches += 1
        else:
            micro_batch_loss = model(input_ids, labels=labels)
            if isinstance(micro_batch_loss, CausalLMOutputWithPast):
                micro_batch_loss = micro_batch_loss.loss
            accelerator.backward(micro_batch_loss)
            if grad_clip is not None:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            micro_batches = 0
            optimizer_steps += 1
            wandb.log({
                "micro_batch_loss": micro_batch_loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "effective_batch_size": gradient_accumulation_steps * micro_batch_size,
                "tokens": index * 512 * micro_batch_size,
                "optimizer_steps": optimizer_steps
            })
            # take 16 steps at each effective batch size before increasing it (to speed initial learning)
            if gradient_accumulation_steps < (max_batch_size // micro_batch_size) and optimizer_steps % steps_per_batch_size == 0:
                gradient_accumulation_steps += 1
        
        if (index + 1) % save_every == 0:
            print("Saving checkpoint...")
            accelerator.save_state(ckpt_dir)
        
        scheduler.step()    
    
if __name__ == "__main__":
    fire.Fire(train)
