import argparse
import gc

import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import TripletEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

import os
import torch.distributed as dist


def get_save_name(model_name, output_base):
    name = model_name.replace('/', '_')
    return os.path.join(output_base, 'outputs', name)


def get_run_name(model_name, run_alias):
    model = model_name.replace('/', '_')
    return f'{model}_{run_alias}'


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


def print_gpu_memory(label=""):
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"[{label}] GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def truncate_dataset(example):
    for field in ['anchor', 'positive', 'negative']:
        if field in example and example[field]:
            example[field] = example[field][:MAX_CHARS]
    return example


def run(arguments):
    print_gpu_memory("Start")
    model = SentenceTransformer(arguments.model_name, trust_remote_code=True)

    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'model_max_length'):
        model.tokenizer.model_max_length = MAX_TOKENS
        print(f"Set tokenizer max_length to {MAX_TOKENS}")
    print_gpu_memory("After model load")

    loss = MultipleNegativesRankingLoss(model)
    print_gpu_memory("After loss init")

    train_dataset = load_dataset(arguments.training_file_path, data_files=arguments.training_file)
    eval_dataset = load_dataset(arguments.eval_file_path, data_files=arguments.eval_file)

    print(f"\nTruncating texts to {MAX_TOKENS} tokens (~{MAX_CHARS} characters)...")
    train_dataset = train_dataset.map(truncate_dataset, batched=False)
    eval_dataset = eval_dataset.map(truncate_dataset, batched=False)
    print_gpu_memory("After dataset load")

    save_name = get_save_name(arguments.model_name, arguments.output_base)
    run_name = get_run_name(arguments.model_name, arguments.run_alias)

    args = SentenceTransformerTrainingArguments(
        output_dir=save_name,
        num_train_epochs=arguments.epochs,
        per_device_train_batch_size=arguments.batch_size,
        per_device_eval_batch_size=arguments.eval_batch_size,
        learning_rate=arguments.learning_rate,
        warmup_ratio=0.1,
        fp16=False,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        gradient_accumulation_steps=4,
        run_name=run_name,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset['train']['anchor'],
        positives=eval_dataset['train']['positive'],
        negatives=eval_dataset['train']['negative'],
        batch_size=arguments.eval_batch_size,
        show_progress_bar=True,
    )

    print_gpu_memory("After evaluator init")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset['train'],
        eval_dataset=eval_dataset['train'],
        loss=loss,
        evaluator=dev_evaluator,
    )

    print_gpu_memory("After trainer init")
    clear_memory()
    print_gpu_memory("After clear_memory")
    trainer.train()

    final_model_path = os.path.join(arguments.output_base, 'models', run_name, 'final')
    os.makedirs(final_model_path, exist_ok=True)
    trainer.save_model(final_model_path)

    if not dist.is_initialized() or dist.get_rank() == 0:
        dev_evaluator(model)


if __name__ == '__main__':
    MAX_TOKENS = 4096
    MAX_CHARS = MAX_TOKENS * 4  # ~4 chars per token = 16,384 characters

    parser = argparse.ArgumentParser(description='Dense sentence transformer fine-tuning')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--training_file_path', type=str, required=True)
    parser.add_argument('--eval_file_path', type=str, required=True)
    parser.add_argument('--training_file', type=str, required=True)
    parser.add_argument('--eval_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--run_alias', type=str, required=True)
    parser.add_argument('--output_base', type=str,
                        default=os.environ.get('MODEL_DIR', '/scratch/hpc/41/dolamull/uk-case-retrieval'),
                        help='Base directory on scratch for checkpoints and final models')
    arguments = parser.parse_args()

    run(arguments)
