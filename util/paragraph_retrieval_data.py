import json

import pandas as pd

from util.retrieval_utils import filter_paragraph_citations, remove_duplicates, load_document


def create_data_split(annotation_file, mapping):

    dataset = []

    with open(annotation_file, 'r') as file:
        annotation = json.load(file)
        for document in annotation:
            paragraph_citations = filter_paragraph_citations(document)
            paragraph_citations = remove_duplicates(paragraph_citations)

            source_document = load_document(document['case'], mapping)
            source_document_text = '\n'.join(source_document['full_text'])
            for citation in paragraph_citations:
                reference_document = load_document(citation['citation'], mapping)
                for paragraph_number in citation['paragraphs']:
                    paragraph_key = paragraph_number + '.'
                    if paragraph_key in source_document['sequence']:
                        reference_paragraph = source_document['paragraphs'][paragraph_key]
                        dataset.append({'source':source_document_text,'reference':reference_paragraph,'score':1})

def load_mappings():
    with open('data/mapping/mapping.json', 'r') as file:
        return json.load(file)


def run():
    mapping = load_mappings()

    train_data = 'data/annotation/training.json'
    dev_data = 'data/annotation/dev.json'
    test_data = 'data/annotation/test.json'

    create_data_split(train_data, mapping)


if __name__ == '__main__':
    run()
