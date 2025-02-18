import argparse
import logging
import random
import traceback
from datetime import datetime

import pandas as pd
from datasets import load_dataset, Dataset
from pandas.core.interchange.dataframe_protocol import DataFrame

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import (
    BinaryClassificationEvaluator,
    InformationRetrievalEvaluator,
    ParaphraseMiningEvaluator,
    SequentialEvaluator,
    TripletEvaluator, EmbeddingSimilarityEvaluator
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments

from util.retrieval_utils import load_document, load_mappings, load_paragraph_dataset


def run():
    # Set the log level to INFO to get more information
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

    train_data_path = 'data/paragraph_retrieval_data/p_ret_training.tsv'
    dev_data_path = 'data/paragraph_retrieval_data/p_ret_dev.tsv'
    test_data_path = 'data/paragraph_retrieval_data/p_ret_test.tsv'
    train_dataset = load_paragraph_dataset(train_data_path)
    eval_dataset = load_paragraph_dataset(dev_data_path)
    test_dataset = load_paragraph_dataset(test_data_path)

    model = SentenceTransformer(
        arguments.model_name
    )
    # 4. Define a loss function
    loss = MultipleNegativesRankingLoss(model)

    model_path = arguments.model_name.replace('/','_')
    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=f"outputs/{model_path}",
        # Optional training parameters:
        num_train_epochs=arguments.epochs,
        per_device_train_batch_size=arguments.batch_size,
        per_device_eval_batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=50,
        run_name=f"document_retrieval_uk_{model_path}",  # Will be used in W&B if `wandb` is installed
    )

    scores = [1.0] * len(eval_dataset["anchor"])  # All pairs have maximum similarity (score 1.0)
    # 6. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset["anchor"],
        sentences2=eval_dataset["positive"],
        scores=scores,
        name="embedding-similarity-evaluator"
    )
    dev_evaluator(model)

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()

    # (Optional) Evaluate the trained model on the test set
    scores = [1.0] * len(test_dataset["anchor"])
    test_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=test_dataset["anchor"],
        sentences2=test_dataset["positive"],
        scores=scores,
        name="embedding-similarity-test-evaluator"
    )
    test_evaluator(model)

    # 8. Save the trained model
    model.save_pretrained(arguments.save_path)


if __name__ == '__main__':
    mappings = load_mappings()
    parser = argparse.ArgumentParser(
        description='''sentence transformer arguments''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--batch_size', type=int, default=4, required=False, help='batch_size')
    parser.add_argument('--epochs', type=int, default=3, required=False, help='epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, required=False, help='learning_rate')
    parser.add_argument('--save_path', type=str, default='models/uk-case-retrieval', required=False,
                        help='model save path')
    arguments = parser.parse_args()
    run()
