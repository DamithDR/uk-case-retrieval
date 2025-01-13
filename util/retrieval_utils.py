import json

import pandas as pd
from datasets import Dataset


def filter_paragraph_citations(document):
    filtered_paragraph_citations = [
        citation for citation in document['paragraph_citations']
        if citation['paragraphs']
    ]
    return filtered_paragraph_citations


def remove_duplicates(paragraph_citations):
    seen = set()
    unique_citations = []
    for citation in paragraph_citations:
        # Create a hashable identifier for each entry
        identifier = (citation['citation'], tuple(citation['paragraphs']))
        if identifier not in seen:
            seen.add(identifier)
            unique_citations.append(citation)
    return unique_citations


def load_document(case_citation, mapping):
    file_path = mapping.get(case_citation)
    if file_path is None:
        print(f'No file for citation : {case_citation}')
        return None
    else:
        with open('data/raw/files/' + file_path, 'r', encoding='utf-8') as file:
            return json.load(file)


def load_mappings():
    with open('data/mapping/mapping.json', 'r') as file:
        return json.load(file)


def map_paragraph(citation):
    split = citation.split('#')
    if len(split) < 2:
        print(citation)
    cite = citation.split('#')[0]
    para = citation.split('#')[1]
    mappings = load_mappings()
    document = load_document(cite, mappings)
    paragraph_num_combinations = [para, para + '.', '(' + para + ')']
    paragraph = ''
    for combination in paragraph_num_combinations:
        if combination in document['sequence']:
            paragraph = document['paragraphs'][combination]['paragraph']
            break

    return paragraph


# Load the TSV file
def if_lexically_similar(anchor_para, positive_para):
    anchor_para_words = set(anchor_para.split(' '))
    positive_para_words = set(positive_para.split(' '))

    common_words = anchor_para_words.intersection(positive_para_words)

    similarity_ratio = len(common_words) / len(anchor_para_words)

    return similarity_ratio >= 0.8


def get_prev_para_num(anchor):
    source_para_num = anchor.split('#')[1]
    source_para_num = source_para_num.replace('.', '').replace('(', '').replace(')', '').replace('“', '')
    source_para_num = int(source_para_num)
    source_para_num -= 1
    return source_para_num


def map_paragraphs(anchor, positive):
    anchor_para = map_paragraph(anchor)
    positive_para = map_paragraph(positive)

    if not if_lexically_similar(anchor_para, positive_para):
        return anchor_para, positive_para, 0
    else:  # get the previous para if the content are same
        source_para_num = get_prev_para_num(anchor)

        if source_para_num >= 1:
            anchor = anchor.split('#')[0] + "#" + str(source_para_num)
            anchor_para = map_paragraph(anchor)
            return anchor_para, positive_para, 1
        else:
            return None, None, 0


def load_paragraph_dataset(tsv_file):
    # Read the TSV file into a pandas DataFrame
    data = pd.read_csv(tsv_file, sep="\t")
    data = data.dropna()

    anchor_list = []
    positive_list = []
    quote_count = 0
    for anchor, positive in zip(data['anchor'], data['positive']):
        anchor_para, positive_para, quoted = map_paragraphs(anchor, positive)
        quote_count += quoted
        if anchor_para is not None and positive_para is not None:
            anchor_list.append(anchor_para)
            positive_list.append(positive_para)
    df = pd.DataFrame({'anchor': anchor_list, 'positive': positive_list})
    dataset = Dataset.from_pandas(df)
    print(f'total quoted count in {tsv_file} = {quote_count}')
    return dataset
