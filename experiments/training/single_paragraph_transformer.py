import argparse
import gc

import torch
from datasets import load_dataset
from sentence_transformers import SparseEncoder, SparseEncoderTrainingArguments, SparseEncoderTrainer, \
    SentenceTransformer
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator, SparseTripletEvaluator
from sentence_transformers.sparse_encoder.losses import SpladeLoss, SparseMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from accelerate import Accelerator

import os, torch.distributed as dist


def get_save_name(model_name):
    name = model_name.replace('/', '_')
    return f'outputs/{name}/'


def get_run_name(model_name):
    model = model_name.replace('/', '_')
    return f'{model}_{arguments.run_alias}'

def clear_memory():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()

def setup_ddp():
    """Initialize torch.distributed and return the local GPU rank."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"[Process {os.getpid()}] Using GPU {local_rank}")
        return local_rank
    else:
        print("Running in single-GPU mode.")
        return 0

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def run(arguments):
    # local_rank = setup_ddp()

    # Load a model to train/finetune
    model = SparseEncoder(arguments.model_name)
    # model = model.to(local_rank)

    # # Wrap in DDP if multi-GPU
    # if torch.distributed.is_initialized():
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[local_rank], output_device=local_rank
    #     )


    accelerator = Accelerator(
        gradient_accumulation_steps=2,
        mixed_precision='fp16',  # Use mixed precision training
        cpu=False,
    )

    # Initialize the SpladeLoss with a SparseMultipleNegativesRankingLoss
    # This loss requires pairs of related texts or triplets
    loss = SpladeLoss(
        model=model,
        loss=SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=5e-5,  # Weight for query loss
        document_regularizer_weight=3e-5,
    )

    # Load an example training dataset that works with our loss function:
    train_dataset = load_dataset(arguments.training_file_path, data_files=arguments.training_file)
    eval_dataset = load_dataset(arguments.training_file_path, data_files=arguments.eval_file)
    print(train_dataset)

    save_name = get_save_name(arguments.model_name)
    run_name = get_run_name(arguments.model_name)

    args = SparseEncoderTrainingArguments(
        # Required parameter:
        output_dir=save_name,
        # Optional training parameters:
        num_train_epochs=arguments.epochs,
        per_device_train_batch_size=arguments.batch_size,
        per_device_eval_batch_size=arguments.eval_batch_size,
        learning_rate=arguments.learning_rate,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        gradient_accumulation_steps=2,
        run_name=run_name,  # Will be used in W&B if `wandb` is installed
    )

    # 6. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = SparseTripletEvaluator(anchors=eval_dataset['train']['anchor'],
                                           positives=eval_dataset['train']["positive"],
                                           negatives=eval_dataset['train']["negative"],
                                           batch_size=arguments.eval_batch_size, show_progress_bar=True)

    # 7. Create a trainer & train
    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )

    # Prepare everything with accelerator
    model, trainer = accelerator.prepare(model, trainer)

    clear_memory()
    trainer.train()

    if accelerator.is_main_process:
        dev_evaluator(model)

    # 9. Save the trained model
    model.save_pretrained(f"models/{run_name}/final")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''sentence transformer arguments''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--training_file_path', type=str, required=True, help='training_file_path')
    parser.add_argument('--eval_file_path', type=str, required=True, help='eval_file_path')
    parser.add_argument('--training_file', type=str, required=True, help='training_file')
    parser.add_argument('--eval_file', type=str, required=True, help='eval_file')
    parser.add_argument('--batch_size', type=int, default=16, required=False, help='batch_size')
    parser.add_argument('--eval_batch_size', type=int, default=16, required=False, help='eval_batch_size')
    parser.add_argument('--epochs', type=int, default=3, required=False, help='epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, required=False, help='learning_rate')
    parser.add_argument('--run_alias', type=str, required=True, help='run_alias')
    arguments = parser.parse_args()

    run(arguments)
