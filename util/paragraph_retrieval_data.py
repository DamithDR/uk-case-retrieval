import json

import pandas as pd

from util.retrieval_utils import load_mappings, load_document


def paragraph_retrievable(citation, paragraph):
    document = load_document(citation, mapping)
    paragraph_num_combinations = [paragraph, paragraph + '.', '(' + paragraph + ')']
    for seq in document['sequence']:
        if seq in paragraph_num_combinations:
            return True
    return False


def create_data_permutations(annotation_file, save_name):
    dataset = []

    with open(annotation_file, 'r') as file:
        annotation = json.load(file)
        for document in annotation:
            source_citation = document['case']
            para_citations = document['paragraph_citations']
            for citation in para_citations:
                if citation['citation']['citation'] in mapping.keys() and len(
                        citation['citation']['paragraphs']) > 0:
                    local_source_citation = source_citation.replace('\n','').strip() + "#" + citation["para"]
                    for paragraph in citation['citation']['paragraphs']:

                        if paragraph_retrievable(citation['citation']['citation'], str(paragraph)):
                            local_destination_citation = citation['citation']['citation'].replace('\n','').strip() + "#" + str(paragraph)
                            dataset.append({'anchor': local_source_citation, 'positive': local_destination_citation})
    df = pd.DataFrame(dataset)
    df.to_csv('data/paragraph_retrieval_data/' + save_name, sep='\t', index=False)


def run():
    train_data = 'data/annotation/training.json'
    dev_data = 'data/annotation/dev.json'
    test_data = 'data/annotation/test.json'

    create_data_permutations(train_data, 'p_ret_training.tsv')
    create_data_permutations(dev_data, 'p_ret_dev.tsv')
    create_data_permutations(test_data, 'p_ret_test.tsv')


if __name__ == '__main__':
    mapping = load_mappings()
    run()
