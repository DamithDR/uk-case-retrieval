import json

from util.retrieval_utils import load_mappings


def count_available_paras(dataset, available_cases):
    para_available_citations = 0
    single_para_citations = 0
    for data in dataset:

        for paragraph in data['paragraph_citations']:
            citation = paragraph['citation']
            if citation['citation'].replace('\n', '').strip() in available_cases:
                if len(citation['paragraphs']) > 0:
                    para_available_citations += 1
                if len(citation['paragraphs']) == 1:
                    single_para_citations += 1
    return para_available_citations, single_para_citations


if __name__ == '__main__':
    with open("data/annotation/training.json", "r", encoding="utf-8") as f:
        training = json.load(f)

    with open("data/annotation/dev.json", "r", encoding="utf-8") as f:
        dev = json.load(f)

    with open("data/annotation/test.json", "r", encoding="utf-8") as f:
        test = json.load(f)

    mappings = load_mappings()

    print(f'paragraphs available citations TEST : {count_available_paras(test, set(mappings.keys()))}')
    print(f'paragraphs available citations DEV : {count_available_paras(dev, set(mappings.keys()))}')
    print(f'paragraphs available citations TRAIN : {count_available_paras(training, set(mappings.keys()))}')
