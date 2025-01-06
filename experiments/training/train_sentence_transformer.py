"""
This scripts demonstrates how to train a sentence embedding model for Information Retrieval.

As dataset, we use Quora Duplicates Questions, where we have pairs of duplicate questions.

As loss function, we use MultipleNegativesRankingLoss. Here, we only need positive pairs, i.e., pairs of sentences/texts that are considered to be relevant. Our dataset looks like this (a_1, b_1), (a_2, b_2), ... with a_i / b_i a text and (a_i, b_i) are relevant (e.g. are duplicates).

MultipleNegativesRankingLoss takes a random subset of these, for example (a_1, b_1), ..., (a_n, b_n). a_i and b_i are considered to be relevant and should be close in vector space. All other b_j (for i != j) are negative examples and the distance between a_i and b_j should be maximized. Note: MultipleNegativesRankingLoss only works if a random b_j is likely not to be relevant for a_i. This is the case for our duplicate questions dataset: If a sample randomly b_j, it is unlikely to be a duplicate of a_i.


The model we get works well for duplicate questions mining and for duplicate questions information retrieval. For question pair classification, other losses (like OnlineConstrativeLoss) work better.
"""
import argparse
import logging
import random
import traceback
from datetime import datetime

import pandas as pd
from datasets import load_dataset, Dataset

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

from util.retrieval_utils import load_document, load_mappings


# Load the TSV file
def load_custom_dataset(tsv_file, citation_mapping):
    # Read the TSV file into a pandas DataFrame
    data = pd.read_csv(tsv_file, sep="\t")

    # Replace citations with actual content using the mapping
    def map_citation(citation):
        document = load_document(citation, citation_mapping)
        return '\n'.join(document['full_text'])

    data['anchor'] = data['anchor'].map(map_citation)
    data['positive'] = data['positive'].map(map_citation)

    dataset = Dataset.from_pandas(data)

    return dataset


def run(arguments):
    # Set the log level to INFO to get more information
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    mappings = load_mappings()

    train_data_path = 'data/document_retrieval_data/d_ret_training.tsv'
    dev_data_path = 'data/document_retrieval_data/d_ret_dev.tsv'
    test_data_path = 'data/document_retrieval_data/d_ret_test.tsv'
    train_dataset = load_custom_dataset(train_data_path, mappings)
    eval_dataset = load_custom_dataset(dev_data_path, mappings)
    test_dataset = load_custom_dataset(test_data_path, mappings)

    model = SentenceTransformer(
        arguments.model_name
    )
    # 4. Define a loss function
    loss = MultipleNegativesRankingLoss(model)

    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir="models/mpnet-base-all-nli-triplet",
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
        logging_steps=100,
        run_name="document_retrieval_uk",  # Will be used in W&B if `wandb` is installed
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
    parser = argparse.ArgumentParser(
        description='''sentence transformer arguments''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--batch_size', type=int, default=4, required=False, help='batch_size')
    parser.add_argument('--epochs', type=int, default=1, required=False, help='epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, required=False, help='learning_rate')
    parser.add_argument('--save_path', type=str, default='models/uk-case-retrieval', required=False,
                        help='model save path')
    args = parser.parse_args()
    run(args)
    run(args)
