import argparse
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers.utils.import_utils import candidates

from util.eval_utils import sort_by_numbers_desc, f1_at_k, precision_at_k, recall_at_k, mean_average_precision
from util.retrieval_utils import load_paragraph_dataset, map_paragraph


def remove_query_from_tests(query):
    keys = []
    embeddings = []
    for candidate, candidate_embedding in zip(candidates, candidate_embeddings):
        if candidate != query:
            keys.append(candidate)
            embeddings.append(candidate_embedding)
    return keys, embeddings


def calculate_similarity(query):
    query_para = map_paragraph(query)
    keys, local_candidate_embeddings = remove_query_from_tests(query)
    query_embeddings = model.encode(query_para)

    similarity_scores = model.similarity(query_embeddings, local_candidate_embeddings)

    return {'keys': keys, 'similarity': similarity_scores}


def output_results(MAP, f1_final, p_final, r_final):
    MAP = round(MAP, 2)
    f1_final = [str(round(f1, 2)) for f1 in f1_final]
    p_final = [str(round(p, 2)) for p in p_final]
    r_final = [str(round(r, 2)) for r in r_final]

    results_file_name = 'results/new_results.csv'
    if not os.path.exists(results_file_name):
        with open(results_file_name, 'a') as f:
            f.write("Model,Dataset,Metric,MAP")
            for number in numbers:
                f.write(f',k_{number}')
            f.write('\n')
    with open(results_file_name, 'a') as f:
        f1_results = ",".join(f1_final)
        p_results = ",".join(p_final)
        r_results = ",".join(r_final)
        f.write(f'{args.model_name},F1,{MAP},{f1_results}\n')
        f.write(f'{args.model_name},Precision,{MAP},{p_results}\n')
        f.write(f'{args.model_name},Recall,{MAP},{r_results}\n')


def run():
    results_dict = {}
    gold = []
    for anchor, test in zip(test_dataset['anchor'], test_dataset['positive']):
        gold.append([test])
        results_dict[anchor] = calculate_similarity(anchor)

    predictions = []

    f1_values = {}
    p_values = {}
    r_values = {}

    print(f"eval k values = {numbers}")

    for anchor, citations in zip(test_dataset['anchor'], gold):
        results = results_dict[anchor]
        values, labels = sort_by_numbers_desc(results['similarity'], results['keys'])
        predictions.append(labels)

        for number in numbers:
            if number not in f1_values.keys():
                f1_values[number] = []
                p_values[number] = []
                r_values[number] = []
            f1_values[number].append(f1_at_k(labels, citations, number))
            p_values[number].append(precision_at_k(labels, citations, number))
            r_values[number].append(recall_at_k(labels, citations, number))

    MAP = mean_average_precision(predictions, gold)
    f1_final = []
    p_final = []
    r_final = []
    for number in numbers:
        f1_final.append(np.mean(f1_values[number]).item())
        p_final.append(np.mean(p_values[number]).item())
        r_final.append(np.mean(r_values[number]).item())

    output_results(MAP, f1_final, p_final, r_final)
    # return MAP, f1_final, p_final, r_final


if __name__ == '__main__':
    numbers = list(range(1, 5)) + list(range(10, 25, 5)) + [50] + [100]

    parser = argparse.ArgumentParser(
        description='''sentence transformer arguments''')
    parser.add_argument('--model_path', type=str, required=True, help='model_path')
    parser.add_argument('--model_name', type=str, required=True, help='model_name')

    args = parser.parse_args()
    model = SentenceTransformer(args.model_path)

    test_data_path = 'data/paragraph_retrieval_data/p_ret_test.tsv'
    test_dataset = load_paragraph_dataset(test_data_path)
    candidates = test_dataset['positive']
    candidate_paras = [map_paragraph(candidate) for candidate in candidates]
    candidate_embeddings = model.encode(candidate_paras)
    run()
