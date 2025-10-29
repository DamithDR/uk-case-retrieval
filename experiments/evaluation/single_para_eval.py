import argparse
import os

import numpy as np
from datasets import load_dataset
from sentence_transformers import SparseEncoder, util
from tqdm import tqdm

from util.eval_utils import recall_at_k, precision_at_k, f1_at_k, mean_average_precision


def output_results(MAP, f1_final, p_final, r_final):
    MAP = round(MAP, 2)
    f1_final = [str(round(f1, 2)) for f1 in f1_final]
    p_final = [str(round(p, 2)) for p in p_final]
    r_final = [str(round(r, 2)) for r in r_final]

    results_file_name = f'results/{arguments.run_alias}_results.csv'
    if not os.path.exists(results_file_name):
        with open(results_file_name, 'a') as f:
            f.write("Model,Metric,MAP")
            for k in k_values:
                f.write(f',k_{k}')
            f.write('\n')
    with open(results_file_name, 'a') as f:
        f1_results = ",".join(f1_final)
        p_results = ",".join(p_final)
        r_results = ",".join(r_final)
        f.write(f'{arguments.model_name},F1,{MAP},{f1_results}\n')
        f.write(f'{arguments.model_name},Precision,{MAP},{p_results}\n')
        f.write(f'{arguments.model_name},Recall,{MAP},{r_results}\n')


def eval_at_k(queries, query_embeddings, corpus, corpus_embeddings, positives, model):
    results = util.semantic_search(query_embeddings, corpus_embeddings, top_k=100, score_function=model.similarity)

    f1_values = {}
    p_values = {}
    r_values = {}
    MAP = 0.0
    pred_lists = []
    positive_lists = []
    for query_id, query in tqdm(enumerate(queries)):
        predictions = []

        for res in results[query_id]:
            corpus_id, score = res.values()
            predicted_paragraph = corpus[corpus_id]
            predictions.append(predicted_paragraph)
        pred_lists.append(predictions)

        positive = [positives[query_id]]
        positive_lists.append(positive)
        for k in k_values:

            if k not in f1_values.keys():
                f1_values[k] = []
                p_values[k] = []
                r_values[k] = []
            f1_values[k].append(f1_at_k(predictions, positive, k))
            p_values[k].append(precision_at_k(predictions, positive, k))
            r_values[k].append(recall_at_k(predictions, positive, k))

        MAP = mean_average_precision(pred_lists, positive_lists)
    f1_final = []
    p_final = []
    r_final = []
    for k in k_values:
        f1_final.append(np.mean(f1_values[k]).item())
        p_final.append(np.mean(p_values[k]).item())
        r_final.append(np.mean(r_values[k]).item())
    output_results(MAP, f1_final, p_final, r_final)


def run(arguments):
    # 1. Load my trained SparseEncoder model
    model = SparseEncoder(arguments.model_name)

    candidates = load_dataset(arguments.candidates_file_path, data_files=arguments.candidates_file)
    gold_data = load_dataset(arguments.gold_file_path, data_files=arguments.gold_file)
    corpus = candidates['train']['candidate']
    queries = gold_data['train']['query']
    positives = gold_data['train']['positive']

    corpus_embeddings = model.encode(corpus)
    query_embeddings = model.encode(queries)

    eval_at_k(queries, query_embeddings, corpus, corpus_embeddings, positives, model)


if __name__ == '__main__':
    k_values = [1] + [i for i in range(5, 51, 5)]
    parser = argparse.ArgumentParser(
        description='''sentence transformer arguments''')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')
    parser.add_argument('--candidates_file_path', type=str, required=True, help='candidates_file_path')
    parser.add_argument('--gold_file_path', type=str, required=True, help='gold_file_path')
    parser.add_argument('--candidates_file', type=str, required=True, help='candidates_file')
    parser.add_argument('--gold_file', type=str, required=True, help='gold_file')

    parser.add_argument('--run_alias', type=str, required=True, help='run_alias')
    arguments = parser.parse_args()
    run(arguments)
