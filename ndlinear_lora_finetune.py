#!/usr/bin/env python3
"""
accelerate_llama_control.py

A script to fine-tune language models using NdLinear-LoRA with Hugging Face Accelerate.

This script demonstrates how to apply a factorized LoRA technique using the NdLinear
layer and train it on a specified dataset from the Hugging Face Hub. It handles
model and tokenizer loading, dataset processing, training, and evaluation.

Example usage:
    python accelerate_llama_control.py \
            --model_name "Qwen/Qwen3-1.7B-Base" \
            --dataset "lmms-lab/Math10K" \
            --output_dir "./output_qwen3_1.7B_math10k_ndlinear_lora" \
            --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
            --lora_alpha 1 \
            --epochs 2 \
            --batch_size 1 \
            --learning_rate 1e-4 \
            --max_length 512 \
            --seed 42
"""

import argparse
import os
import math
import logging
import time
import random
import json
from pathlib import Path
import traceback
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from ndlinear import NdLinear
from accelerate import Accelerator
from accelerate.utils import set_seed
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeneralTextDataset(Dataset):
    """
    Dataset for handling text data from Hugging Face Hub.
    It expects a specific structure in the dataset items. The supported formats are, in order of priority:
    1. A dictionary with 'instruction', 'output', and 'answer' keys.
    2. A dictionary with a single 'text' key.
    """
    def __init__(self, dataset, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer pad_token set to eos_token: {self.tokenizer.eos_token}")

        logger.info("Processing dataset items...")
        for item in tqdm(dataset, desc="Tokenizing dataset"):
            prompt_text = ""
            target_text_for_loss = ""
            full_text_for_tokenizer = ""

            # Preferred format: Instruction, Solution, Final Answer
            if "instruction" in item and "output" in item and "answer" in item:
                instruction_content = str(item["instruction"]).strip()
                output_content = str(item["output"]).strip()
                answer_content = str(item["answer"]).strip()
                prompt_text = f"### Instruction:\n{instruction_content}\n\n### Solution:\n"
                target_text_for_loss = f"{output_content}\n\n### Final Answer:\n\boxed{{{answer_content}}}"
                full_text_for_tokenizer = prompt_text + target_text_for_loss

            # Fallback for general text-based datasets
            elif "text" in item:
                full_text_content = str(item["text"]).strip()
                prompt_text = ""  # No prompt, learn the whole sequence
                target_text_for_loss = full_text_content
                full_text_for_tokenizer = full_text_content
            
            else:
                logger.warning(f"Skipping item due to unrecognized format. Expected keys like ('instruction', 'output', 'answer') or ('text'). Found: {list(item.keys())}")
                continue

            # Add EOS token if not present
            if not full_text_for_tokenizer.endswith(self.tokenizer.eos_token):
                full_text_for_tokenizer += self.tokenizer.eos_token

            # Tokenize the full text
            full_tokenized = self.tokenizer(
                full_text_for_tokenizer,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_attention_mask=True
            )

            # Determine prompt length for masking
            prompt_tokens_len = len(self.tokenizer(prompt_text, truncation=True, max_length=self.max_length)['input_ids'])

            input_ids = full_tokenized['input_ids']
            attention_mask = full_tokenized['attention_mask']
            
            # Create labels, masking the prompt part
            labels = [-100] * len(input_ids)
            target_start_index = min(prompt_tokens_len, len(input_ids))
            for i in range(target_start_index, len(input_ids)):
                labels[i] = input_ids[i]

            # Pad sequences to max_length
            padding_length = self.max_length - len(input_ids)
            if padding_length > 0:
                input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
                attention_mask.extend([0] * padding_length)
                labels.extend([-100] * padding_length)

            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def find_linear_modules(model, target_modules):
    """Finds all nn.Linear modules whose names contain any of the target substrings."""
    linear_modules = {}
    target_set = set(target_modules)
    for name, module in model.named_modules():
        if any(target in name for target in target_set) and isinstance(module, nn.Linear):
            linear_modules[name] = module
    return linear_modules

def find_factor(n):
    """Finds the most balanced integer factors for n."""
    for i in range(int(n ** 0.5), 0, -1):
        if n % i == 0:
            return (i, n // i)
    return (1, n)

class NdLinearLoRA(nn.Module):
    """The NdLinear-LoRA adapter layer."""
    def __init__(self, d_in, d_out, alpha=1.0, dropout=0.05):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.in_factors = find_factor(d_in)
        self.out_factors = find_factor(d_out)
        self.adapter = NdLinear(
            input_dims=self.in_factors,
            hidden_size=self.out_factors,
            transform_outer=False,
            bias=False
        )
        self.scaling = alpha
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        orig_shape = x.shape  # (B, L, d_in)
        x = self.drop(x).view(-1, *self.in_factors)
        y = self.adapter(x).view(*orig_shape[:-1], self.d_out)
        return y * self.scaling

class LinearWithNdLinearLoRA(nn.Module):
    """A nn.Linear layer wrapped with the NdLinear-LoRA adapter."""
    def __init__(self, base_layer, alpha=1.0, dropout=0.05):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.adapter = NdLinearLoRA(
            d_in=self.base_layer.in_features,
            d_out=self.base_layer.out_features,
            alpha=alpha,
            dropout=dropout
        )

    def forward(self, x):
        return self.base_layer(x) + self.adapter(x)

def apply_ndlinear_lora(model, target_modules, alpha, dropout):
    """Applies NdLinear-LoRA to the specified model layers."""
    device = next(model.parameters()).device
    wrapped_modules = {}
    linear_modules = find_linear_modules(model, target_modules)
    logger.info(f"Applying NdLinear-LoRA to {len(linear_modules)} modules...")
    for name in linear_modules.keys():
        logger.info(f"  - Targeting module: {name}")

    total_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, module in linear_modules.items():
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        wrapped = LinearWithNdLinearLoRA(module, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, wrapped)
        wrapped_modules[name] = wrapped

    total_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"NdLinear-LoRA trainable parameters: {total_params_after - total_params_before:,}")
    return model, wrapped_modules

def train_epoch(model, train_dataloader, optimizer, accelerator, gradient_accumulation_steps, current_epoch, use_wandb, log_interval):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {current_epoch+1}", disable=not accelerator.is_main_process)
    
    for step, batch in enumerate(progress_bar):
        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            
            optimizer.step()
            optimizer.zero_grad()

        # Logging
        total_loss += loss.item() / gradient_accumulation_steps
        if accelerator.is_main_process:
            progress_bar.set_postfix({"loss": total_loss / (step + 1)})
            if use_wandb and (step + 1) % log_interval == 0:
                accelerator.log({"train/step_loss": loss.item() / gradient_accumulation_steps, "epoch": current_epoch, "step": step})

    return total_loss / len(train_dataloader)

def evaluate(model, eval_dataloader, accelerator):
    """Evaluates the model on the provided dataloader."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
            outputs = model(**batch)
            total_loss += accelerator.gather(outputs.loss).mean().item()
    return total_loss / len(eval_dataloader)

def count_parameters(model):
    """Counts trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def run_experiment(args):
    """Main function to run the NdLinear-LoRA fine-tuning experiment."""
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None
    )

    if args.use_wandb and accelerator.is_main_process:
        wandb_kwargs = {"name": args.wandb_run_name} if args.wandb_run_name else {}
        if args.wandb_entity: wandb_kwargs["entity"] = args.wandb_entity
        accelerator.init_trackers(project_name=args.wandb_project, config=vars(args), init_kwargs={"wandb": wandb_kwargs})

    logger.info(f"Accelerator Initialized: {accelerator.state}")
    device = accelerator.device

    # --- Dataset Loading ---
    logger.info(f"Loading dataset: {args.dataset}")
    raw_dataset = load_dataset(args.dataset)
    
    # Split dataset if no validation split exists
    if 'validation' not in raw_dataset.keys():
        logger.info("No validation split found. Creating one from the training set.")
        train_val_split = raw_dataset['train'].train_test_split(test_size=args.validation_split_percentage, seed=args.seed)
        train_dataset = train_val_split['train']
        val_dataset = train_val_split['test']
    else:
        train_dataset = raw_dataset['train']
        val_dataset = raw_dataset['validation']
        
    if args.sample_size:
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(min(args.sample_size, len(train_dataset))))
        val_sample_size = args.sample_size // 10
        val_dataset = val_dataset.shuffle(seed=args.seed).select(range(min(val_sample_size, len(val_dataset))))

    logger.info(f"Using {len(train_dataset)} training examples and {len(val_dataset)} validation examples.")

    # --- Tokenizer and Dataloaders ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    train_data = GeneralTextDataset(train_dataset, tokenizer, max_length=args.max_length)
    val_data = GeneralTextDataset(val_dataset, tokenizer, max_length=args.max_length)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)

    # --- Model Setup and LoRA Application ---
    logger.info(f"Loading base model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=hf_token)
    
    logger.info("Freezing base model parameters...")
    for param in model.parameters():
        param.requires_grad = False
        
    target_modules = args.target_modules.split(",")
    model, wrapped_modules = apply_ndlinear_lora(model, target_modules, alpha=args.lora_alpha, dropout=args.dropout)

    # --- Checkpoint Loading ---
    if args.continue_from_checkpoint and os.path.exists(args.continue_from_checkpoint):
        checkpoint_path = os.path.join(args.continue_from_checkpoint, "pytorch_model.bin")
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading NdLinear-LoRA weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            incompatible_keys = model.load_state_dict(state_dict, strict=False)
            logger.info(f"Checkpoint loaded. Missing: {len(incompatible_keys.missing_keys)}, Unexpected: {len(incompatible_keys.unexpected_keys)}")
        else:
            logger.warning(f"Checkpoint directory specified, but 'pytorch_model.bin' not found in {args.continue_from_checkpoint}.")

    trainable_params, total_params = count_parameters(model)
    logger.info(f"NdLinear-LoRA - Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # --- Training Loop ---
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, accelerator, args.gradient_accumulation_steps, epoch, args.use_wandb, args.log_interval)
        val_loss = evaluate(model, val_dataloader, accelerator)
        
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if args.use_wandb:
                accelerator.log({"train/epoch_loss": train_loss, "val/epoch_loss": val_loss, "epoch": epoch + 1})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                unwrapped_model = accelerator.unwrap_model(model)
                
                # Save model, tokenizer, and config
                unwrapped_model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                
                # Save NdLinear-LoRA specific config
                with open(os.path.join(args.output_dir, "ndlinear_lora_config.json"), "w") as f:
                    wrapped_info = {
                        name: {
                            "in_features": mod.adapter.d_in, "out_features": mod.adapter.d_out,
                            "in_factors": list(mod.adapter.in_factors), "out_factors": list(mod.adapter.out_factors)
                        } for name, mod in wrapped_modules.items()
                    }
                    json.dump({"base_model_name": args.model_name, "wrapped_modules": wrapped_info}, f, indent=2)

                logger.info(f"New best model saved to {args.output_dir} with validation loss: {val_loss:.4f}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f}")
                    break
    
    if args.use_wandb:
        accelerator.end_training()
    logger.info("=== Experiment completed successfully. ===")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model using NdLinear-LoRA with Accelerate.")
    
    # Model and LoRA args
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Hugging Face model name or path.")
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", help="Comma-separated list of module names to apply NdLinear-LoRA.")
    parser.add_argument("--lora_alpha", type=int, default=1, help="LoRA alpha scaling factor.")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate for LoRA layers.")
    
    # Dataset args
    parser.add_argument("--dataset", type=str, required=True, help="Hugging Face dataset ID (e.g., 'lucadiliello/oasst-code-alpaca-20k').")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional: number of examples to use from the dataset for faster runs.")
    parser.add_argument("--validation_split_percentage", type=float, default=0.05, help="Percentage of training data for validation if no 'validation' split exists.")

    # Training args
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients before updating.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision training type.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience in epochs.")
    
    # System and Output args
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the final model and results.")
    parser.add_argument("--continue_from_checkpoint", type=str, default=None, help="Path to a checkpoint directory to continue training from.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    
    # WandB args
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="ndlinear-lora-experiments", help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (defaults to auto-generated).")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team name).")
    parser.add_argument("--log_interval", type=int, default=10, help="Log training step loss to W&B every N steps.")
    
    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main() 

