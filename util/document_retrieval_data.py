import json

import pandas as pd

from util.retrieval_utils import load_mappings


# ['anchor', 'positive']
def create_data_permutations(annotation_file, save_name):
    dataset = []

    with open(annotation_file, 'r') as file:
        annotation = json.load(file)
        for document in annotation:
            source_citation = document['case']
            citations = document['citations']
            for citation in citations:
                if citation in mapping.keys():
                    dataset.append({'anchor': source_citation, 'positive': citation})
    df = pd.DataFrame(dataset)
    df.to_csv('data/document_retrieval_data/' + save_name, sep='\t', index=False)


def run():
    train_data = 'data/annotation/training.json'
    dev_data = 'data/annotation/dev.json'
    test_data = 'data/annotation/test.json'

    create_data_permutations(train_data, 'd_ret_training.tsv')
    create_data_permutations(test_data, 'd_ret_test.tsv')
    create_data_permutations(dev_data, 'd_ret_dev.tsv')


if __name__ == '__main__':
    mapping = load_mappings()
    run()
