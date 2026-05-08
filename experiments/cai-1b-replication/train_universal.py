#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-GPU Fine-tuning with Unsloth - Multi-Architecture Support
Automatically detects and handles Llama, Qwen, and Mistral architectures
"""

import argparse
import logging
import os
# Set environment variables for Unsloth to work in burst mode
os.environ['TMPDIR'] = '/shared/tmp'
os.environ['TEMP'] = '/shared/tmp'
os.environ['TMP'] = '/shared/tmp'
os.environ['HOME'] = '/shared/tmp'
os.environ["WANDB_DIR"] = "/shared/wandb"
os.environ["WANDB_CACHE_DIR"] = "/shared/wandb_cache"
os.environ["TMPDIR"] = "/shared/tmp"
os.makedirs('/shared/tmp/unsloth_offload', exist_ok=True)

# Create the directories
os.makedirs("/shared/wandb", exist_ok=True)
os.makedirs("/shared/wandb_cache", exist_ok=True)
os.environ['UNSLOTH_LOGGING_ENABLED'] = '1'
os.environ["UNSLOTH_DISABLE_TRAINER_PATCHING"] = "1"
os.environ["UNSLOTH_NO_CUDA_EXTENSIONS"] = "1"
os.environ['UNSLOTH_DISABLE_RL_PATCHING'] = '1'

from unsloth import FastLanguageModel
import torch
import wandb
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_model_architecture(tokenizer) -> str:
    """
    Detect model architecture from tokenizer special tokens.
    Returns: 'llama', 'qwen', 'mistral', 'phi', 'gemma', or 'unknown'
    """
    special_tokens_str = str(tokenizer.special_tokens_map)

    if '<|im_start|>' in special_tokens_str or '<|im_end|>' in special_tokens_str:
        return 'qwen'
    elif '<|begin_of_text|>' in special_tokens_str or '<|start_header_id|>' in special_tokens_str:
        return 'llama'
    elif '<|user|>' in special_tokens_str or '<|assistant|>' in special_tokens_str:
        return 'phi'
    elif '<start_of_turn>' in special_tokens_str or '<end_of_turn>' in special_tokens_str:
        return 'gemma'
    elif tokenizer.bos_token == '<s>' and tokenizer.eos_token == '</s>':
        return 'mistral'

    return 'unknown'


def setup_model_and_tokenizer(model_path: str, architecture: str, max_seq_length: int = 1048, device_map: str = "auto"):
    """Load model and tokenizer with Unsloth."""
    logger.info(f"Loading model from: {model_path}")
    logger.info(f"Specified architecture: {architecture.upper()}")
    logger.info(f"Device map: {device_map}")

    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    logger.info(f"Using local rank: {local_rank}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map={"": local_rank}
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    detected = detect_model_architecture(tokenizer)
    if detected != 'unknown' and detected != architecture:
        logger.warning(f"⚠️  Tokenizer suggests {detected.upper()} but you specified {architecture.upper()}")
        logger.warning(f"⚠️  Proceeding with your specification ({architecture.upper()})")
    elif detected == architecture:
        logger.info(f"✅ Architecture verified: {architecture.upper()}")
    else:
        logger.info(f"ℹ️  Could not verify architecture from tokenizer, trusting your specification")

    logger.info(f"Special tokens - BOS: {tokenizer.bos_token}, EOS: {tokenizer.eos_token}, PAD: {tokenizer.pad_token}")

    return model, tokenizer


def setup_lora(model, r: int = 16, lora_alpha: int = 8, lora_dropout: float = 0):
    """Configure and add LoRA adapter to model."""
    logger.info("Setting up LoRA configuration")

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj", "lm_head", "embed_tokens"],
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        temporary_location="",
    )

    logger.info("LoRA adapter added successfully")
    return model


def load_and_prepare_dataset(
    dataset_name: str,
    hf_token: str = None,
    val_fraction: float = 0.1,
    seed: int = 42,
):
    """
    Load the CAI animal welfare dataset.
    Columns: prompt, init_response, critique, revised_response.
    We train on prompt -> revised_response (the compassionate CAI output).
    """
    logger.info(f"Loading dataset: {dataset_name}")

    data = load_dataset(dataset_name, token=hf_token)
    full = data["train"]

    # Filter empty rows
    before = len(full)
    full = full.filter(lambda x: bool(x["prompt"]) and bool(x["revised_response"]))
    logger.info(f"Filtered {before - len(full)} empty rows; {len(full)} remain")

    # Drop columns we don't need for training (keep prompt + revised_response only)
    full = full.remove_columns(["init_response", "critique"])

    split = full.train_test_split(test_size=val_fraction, seed=seed)
    train_src = split["train"]
    val_src   = split["test"]

    logger.info(f"Training samples: {len(train_src)}, Validation samples: {len(val_src)}")
    return train_src, val_src


# ── Chat formatters ───────────────────────────────────────────────────────────
# System prompt reflects the CAI animal-welfare fine-tune purpose.

SYSTEM_PROMPT = (
    "You are a compassionate AI assistant that takes animal welfare seriously. "
    "When asked about topics involving animals, you provide thoughtful responses "
    "that acknowledge animal sentience and promote cruelty-free alternatives where possible."
)


def format_chat_llama(prompt: str, response: str) -> str:
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{response}<|eot_id|>"
    )


def format_chat_qwen(prompt: str, response: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


def format_chat_mistral(prompt: str, response: str) -> str:
    return (
        f"<s>[INST] {SYSTEM_PROMPT}\n\n{prompt} [/INST] {response}</s>"
    )


def format_chat_phi(prompt: str, response: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n{prompt}<|end|>\n"
        f"<|assistant|>\n{response}<|end|>"
    )


def format_chat_gemma(prompt: str, response: str) -> str:
    return (
        f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n{response}<end_of_turn>"
    )


def get_assistant_marker(architecture: str) -> tuple:
    if architecture == 'llama':
        return ("<|start_header_id|>assistant<|end_header_id|>\n\n", {})
    elif architecture == 'qwen':
        return ("<|im_start|>assistant\n", {})
    elif architecture == 'mistral':
        return ("[/INST]", {})
    elif architecture == 'phi':
        return ("<|assistant|>\n", {})
    elif architecture == 'gemma':
        return ("<start_of_turn>model\n", {})
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def tokenize_qa_with_masking(examples, tokenizer, architecture: str, max_length: int = 2048):
    """Tokenize QA pairs with architecture-specific chat template and proper masking."""
    format_map = {
        'llama':   format_chat_llama,
        'qwen':    format_chat_qwen,
        'mistral': format_chat_mistral,
        'phi':     format_chat_phi,
        'gemma':   format_chat_gemma,
    }
    if architecture not in format_map:
        raise ValueError(f"Unknown architecture: {architecture}")
    format_func = format_map[architecture]

    all_texts = []
    for i in range(len(examples["prompt"])):
        inst = examples["prompt"][i]
        resp = examples["revised_response"][i]
        all_texts.append(format_func(inst, resp))

    tokenized_batch = tokenizer(
        all_texts,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        add_special_tokens=False,          # format strings already include BOS/EOS
    )

    assistant_marker, _ = get_assistant_marker(architecture)
    assistant_marker_tokens = tokenizer(assistant_marker, add_special_tokens=False).input_ids

    all_labels = []
    for i in range(len(tokenized_batch["input_ids"])):
        current_input_ids = tokenized_batch["input_ids"][i]
        labels = current_input_ids.copy()

        assistant_idx = -1
        for k in range(len(current_input_ids) - len(assistant_marker_tokens) + 1):
            if current_input_ids[k : k + len(assistant_marker_tokens)] == assistant_marker_tokens:
                assistant_idx = k + len(assistant_marker_tokens)
                break

        if assistant_idx != -1:
            for j in range(assistant_idx):
                labels[j] = -100
        else:
            logger.warning(f"Could not find assistant marker in example {i}, using fallback masking")
            for j in range(len(labels) - 50):
                labels[j] = -100

        for j in range(len(labels)):
            if current_input_ids[j] == tokenizer.pad_token_id:
                labels[j] = -100

        all_labels.append(labels)

    tokenized_batch["labels"] = all_labels
    return tokenized_batch


def train_model(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    architecture: str,
    wandb_project: str,
    wandb_run_name: str,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_steps: int = 500,
    learning_rate: float = 2e-4,
    num_epochs: int = 7,
):
    """Train the model with specified configuration."""
    logger.info("Setting up training")
    logger.info(f"Training with {architecture.upper()} architecture")

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={"architecture": architecture}
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="/shared/cai_model_output",      # checkpoints saved here locally
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        logging_steps=20,                           # log every 20 steps to W&B
        eval_strategy="steps",
        eval_steps=20,
        save_steps=20,
        save_total_limit=4,                         # keep last 4 checkpoints locally
        load_best_model_at_end=True,
        greater_is_better=False,
        bf16=True,
        metric_for_best_model="eval_loss",
        optim="adamw_8bit",
        weight_decay=0.01,
        report_to="wandb",                          # all step metrics go to W&B
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False,
        push_to_hub=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        args=training_args,
        callbacks=[
            # Stop if eval_loss does not improve for 2 consecutive eval steps
            EarlyStoppingCallback(early_stopping_patience=2),
        ],
    )

    logger.info("Starting training...")
    trainer.train()

    last_lrs = trainer.lr_scheduler.get_last_lr()
    logger.info(f"Final learning rates: {[f'{lr:.2e}' for lr in last_lrs]}")

    logger.info("Saving final LoRA adapter model to /shared/cai_model_output/final_model ...")
    final_save_path = "/shared/cai_model_output/final_model"
    trainer.save_model(final_save_path)

    return trainer, final_save_path


def upload_to_huggingface(
    model_path: str,
    hf_repo_name: str,
    hf_token: str,
    max_seq_length: int = 2048
):
    """Load LoRA adapter model, merge weights, and upload merged model to HuggingFace."""
    logger.info(f"Loading LoRA adapter model from: {model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()

    logger.info(f"Uploading merged model to HuggingFace: {hf_repo_name}")
    merged_model.push_to_hub(hf_repo_name, token=hf_token)
    tokenizer.push_to_hub(hf_repo_name, token=hf_token)

    logger.info("✅ Merged model upload complete!")
    logger.info(f"Model available at: {hf_repo_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.1 8B on the CAI animal-welfare dataset"
    )

    # Model
    parser.add_argument('--model-path', required=True,
                        help='Path to base model (local path or HF repo)')
    parser.add_argument('--architecture', type=str, required=True,
                        choices=['llama', 'qwen', 'mistral', 'phi', 'gemma'],
                        help='Model architecture')
    parser.add_argument('--device-map', default='auto')

    # Dataset
    parser.add_argument('--dataset-name',
                        default='CompassioninMachineLearning/cai-animal-harm-sft',
                        help='HuggingFace dataset repo')
    parser.add_argument('--val-fraction', type=float, default=0.1,
                        help='Fraction of data to use for validation')

    # Training hyperparameters
    parser.add_argument('--max-seq-length',              type=int,   default=1048)
    parser.add_argument('--per-device-batch-size',       type=int,   default=4)
    parser.add_argument('--gradient-accumulation-steps', type=int,   default=4)
    parser.add_argument('--max-steps',                   type=int,   default=500)
    parser.add_argument('--learning-rate',               type=float, default=5e-5)
    parser.add_argument('--num-epochs',                  type=int,   default=3)

    # LoRA
    parser.add_argument('--lora-r',       type=int,   default=16)
    parser.add_argument('--lora-alpha',   type=int,   default=8)
    parser.add_argument('--lora-dropout', type=float, default=0)

    # W&B — logs metrics every step, no checkpoint artifacts uploaded
    parser.add_argument('--wandb-project',  default='compassion-in-ml')
    parser.add_argument('--wandb-run-name', required=True)
    parser.add_argument('--wandb-api-key',  required=True)

    # HuggingFace
    parser.add_argument('--hf-repo-name', required=True,
                        help='HF repo to push the final merged model to')
    parser.add_argument('--hf-token', required=True)

    args = parser.parse_args()

    os.environ['WANDB_API_KEY'] = args.wandb_api_key
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Load model
    model, tokenizer = setup_model_and_tokenizer(
        args.model_path, args.architecture, args.max_seq_length, args.device_map
    )

    # LoRA
    model = setup_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    # Move any non-quantized params (lm_head, embed_tokens LoRA) to correct GPU
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    model = model.to(f'cuda:{local_rank}')
    logger.info(f"Moved model to cuda:{local_rank}")

    # Dataset
    train_src, val_src = load_and_prepare_dataset(
        args.dataset_name,
        hf_token=args.hf_token,
        val_fraction=args.val_fraction,
    )

    # Tokenise
    logger.info(f"Tokenizing with {args.architecture.upper()} chat template...")
    train_tokenized = train_src.map(
        lambda x: tokenize_qa_with_masking(x, tokenizer, args.architecture, args.max_seq_length),
        batched=True, batch_size=100, remove_columns=train_src.column_names
    )
    val_tokenized = val_src.map(
        lambda x: tokenize_qa_with_masking(x, tokenizer, args.architecture, args.max_seq_length),
        batched=True, batch_size=100, remove_columns=val_src.column_names
    )

    # Sanity-check tokenisation
    logger.info("=" * 60)
    sample_text = tokenizer.decode(train_tokenized[0]["input_ids"])
    logger.info(f"Sample formatted text:\n{sample_text[:500]}...")
    logger.info("=" * 60)

    # Train
    trainer, final_model_path = train_model(
        model, tokenizer,
        train_tokenized, val_tokenized,
        args.architecture,
        args.wandb_project, args.wandb_run_name,
        args.per_device_batch_size,
        args.gradient_accumulation_steps,
        args.max_steps,
        args.learning_rate,
        args.num_epochs,
    )

    # Merge and push to HF
    logger.info("=" * 60)
    logger.info("UPLOADING MERGED MODEL TO HUGGINGFACE")
    logger.info("=" * 60)
    upload_to_huggingface(
        final_model_path, args.hf_repo_name, args.hf_token, args.max_seq_length
    )

    wandb.finish()

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"✅ Architecture:          {args.architecture.upper()}")
    logger.info(f"✅ Checkpoints saved to:  /shared/cai_model_output/")
    logger.info(f"✅ W&B run:               {args.wandb_project}/{args.wandb_run_name}")
    logger.info(f"✅ Model pushed to HF:    {args.hf_repo_name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
