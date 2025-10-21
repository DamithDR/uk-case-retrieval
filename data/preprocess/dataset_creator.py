import json
import random

import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from tqdm import tqdm

from util.retrieval_utils import load_mappings, load_document


def get_negative_multi_paras(case, position, window_size):
    seq = case['sequence']
    width = window_size // 2
    idx = seq.index(position)

    start = max(0, idx - width)
    end = min(len(seq), idx + width + 1)
    new_sequence = seq.copy()
    del new_sequence[start:end]

    new_index = random.randint(0, len(new_sequence) - width)
    new_start = max(0, new_index - width)
    new_end = min(len(new_sequence), new_index + width + 1)

    paras = [case['paragraphs'][i]['paragraph'] for i in new_sequence[new_start:new_end]]
    return "\n".join(paras) + "\n"


def get_multi_paras(case, position, window_size):
    seq = case['sequence']
    width = window_size // 2
    idx = seq.index(position)

    start = max(0, idx - width)
    end = min(len(seq), idx + width + 1)

    paras = [case['paragraphs'][i]['paragraph'] for i in seq[start:end]]
    return "\n".join(paras) + "\n"


def permute_paragraphs(anchor, anchor_position, positive, positive_positions, window_size, with_negative=False):
    if window_size <= 0:
        raise ValueError('window size should be > 0')

    if positive_positions[0] in positive['paragraphs'].keys():
        positive_position = positive_positions[0]
    elif str(positive_positions[0]) in positive['paragraphs'].keys():
        positive_position = str(positive_positions[0])
    elif f'{positive_positions[0]}.' in positive['paragraphs'].keys():
        positive_position = f'{positive_positions[0]}.'
    else:
        raise KeyError('positive position not available')
    if len(positive['sequence']) > 5:
        if window_size > 1:
            anchor_paragraph = get_multi_paras(anchor, anchor_position, window_size)
            # positive_position = positive_positions[0] if positive_positions[0] in positive[
            #     'paragraphs'].keys() else f'{positive_positions[0]}.'
            positive_paragraph = get_multi_paras(positive, positive_position, window_size)
            negative_paragraph = get_negative_multi_paras(positive, positive_position,
                                                          window_size) if with_negative else ""
        else:
            anchor_paragraph = anchor['paragraphs'][anchor_position]['paragraph']
            # positive_position = f'{positive_positions[0]}.'  # taking only the first paragraph to reduce the conflict
            # positive_position = positive_positions[0] if positive_positions[0] in positive[
            #     'paragraphs'].keys() else f'{positive_positions[0]}.'

            positive_paragraph = positive['paragraphs'][positive_position]['paragraph']
            negative_paragraph = get_negative_multi_paras(positive, positive_position,
                                                          window_size) if with_negative else ""
    else:
        raise KeyError('not enough paragraphs <=5')
    return anchor_paragraph, positive_paragraph, negative_paragraph


# def get_anchor_positive_paragraphs(case, anchor_para, cite_info):
#     anchor_case = load_document(case, mappings)
#     positive_case = load_document(cite_info['citation'].replace('\n', '').strip(), mappings)
#     positive_positions = cite_info['paragraphs']
#     anchor_para_text, positive_para_text, _ = permute_paragraphs(anchor_case, anchor_para, positive_case,
#                                                                  positive_positions, 1, False)
#     anchor_para_multi_text, positive_para_multi_text, _ = permute_paragraphs(anchor_case, anchor_para, positive_case,
#                                                                              positive_positions, 3, False)
#     return anchor_para_text, positive_para_text, anchor_para_multi_text, positive_para_multi_text


def get_anchor_positive_negative_paragraphs(case, anchor_para, cite_info):
    anchor_case = load_document(case, mappings)
    positive_case = load_document(cite_info['citation'].replace('\n', '').strip(), mappings)
    positive_positions = cite_info['paragraphs']
    anchor_para_text, positive_para_text, negative_para_text = permute_paragraphs(anchor_case, anchor_para,
                                                                                  positive_case,
                                                                                  positive_positions, 1, True)
    anchor_para_multi_text, positive_para_multi_text, negative_para_multi_text = permute_paragraphs(anchor_case,
                                                                                                    anchor_para,
                                                                                                    positive_case,
                                                                                                    positive_positions,
                                                                                                    3, True)
    return anchor_para_text, positive_para_text, negative_para_text, anchor_para_multi_text, positive_para_multi_text, negative_para_multi_text


def save_to_file(data, path):
    df = pd.DataFrame(data)
    df.to_csv(path, sep="\t", index=False)


def create_supervised_dataset(dataset, available_cases):
    # gold_citations = {"anchor_citation": [], "gold_citation": []}
    apd = {"anchor": [], "positive": []}  # single para and positive only
    mapd = {"anchor": [], "positive": []}  # multiple paras and positive only
    apnd = {"anchor": [], "positive": [], "negative": []}  # multiple paras and positive and negative
    mapnd = {"anchor": [], "positive": [], "negative": []}  # multiple paras and positive and negative

    for data in tqdm(dataset):
        for paragraph in data['paragraph_citations']:
            citation = paragraph['citation']
            if citation['citation'].replace('\n', '').strip() in available_cases and len(
                    citation['paragraphs']) > 0:
                case = data['case'].replace('\n', '').strip()
                # apt, ppt, apmt, ppmt = get_anchor_positive_paragraphs(case, citation['para'], citation)
                try:
                    apt, ppt, npt, apmt, ppmt, npmt = get_anchor_positive_negative_paragraphs(case, paragraph['para'],
                                                                                              citation)
                    apd['anchor'].append(apt)
                    apd['positive'].append(ppt)
                    mapd['anchor'].append(apmt)
                    mapd['positive'].append(ppmt)

                    apnd['anchor'].append(apt)
                    apnd['positive'].append(ppt)
                    apnd['negative'].append(npt)
                    mapnd['anchor'].append(apmt)
                    mapnd['positive'].append(ppmt)
                    mapnd['negative'].append(npmt)
                except KeyError as ex:
                    continue
    save_to_file(apd, "data/data_splits/training/anchor_positive_W1.tsv")
    save_to_file(mapd, "data/data_splits/training/anchor_positive_W3.tsv")
    save_to_file(apnd, "data/data_splits/training/anchor_positive_negative_W1.tsv")
    save_to_file(mapnd, "data/data_splits/training/anchor_positive_negative_W3.tsv")

    len_apd = len(apd['anchor'])
    print(f'anchor positive training set size {len_apd}')
    len_apnd = len(apnd['anchor'])
    print(f'anchor positive negative training set size {len_apnd}')
    len_mapd = len(mapd['anchor'])
    print(f'multiple anchor positive training set size {len_mapd}')
    len_mapnd = len(mapnd['anchor'])
    print(f'multiple anchor positive negative training set size {len_mapnd}')


def create_candidates_and_queries(dataset, available_cases):
    candidates_1P = {"candidate": []}
    gold_1P = {"query": [], "positive": []}

    candidates_3P = {"candidate": []}
    gold_3P = {"query": [], "positive": []}
    for data in tqdm(dataset):
        for paragraph in data['paragraph_citations']:
            citation = paragraph['citation']
            if citation['citation'].replace('\n', '').strip() in available_cases and len(
                    citation['paragraphs']) > 0:
                case = data['case'].replace('\n', '').strip()
                try:
                    apt, ppt, npt, apmt, ppmt, npmt = get_anchor_positive_negative_paragraphs(case, paragraph['para'],
                                                                                              citation)

                    candidates_1P['candidate'].append(ppt)
                    candidates_1P['candidate'].append(npt)
                    gold_1P['query'].append(apt)
                    gold_1P['positive'].append(ppt)

                    candidates_3P['candidate'].append(ppmt)
                    candidates_3P['candidate'].append(npmt)
                    gold_3P['query'].append(apmt)
                    gold_3P['positive'].append(ppmt)

                except KeyError as ex:
                    continue
    save_to_file(candidates_1P, 'data/data_splits/candidates_1P.tsv')
    save_to_file(gold_1P, 'data/data_splits/gold_1P.tsv')

    save_to_file(candidates_3P, 'data/data_splits/candidates_3P.tsv')
    save_to_file(gold_3P, 'data/data_splits/gold_3P.tsv')

    len_1p_candidates = len(candidates_1P['candidate'])
    print(f'1 paragraph candidates size {len_1p_candidates}')
    len_1p_gold = len(gold_1P['query'])
    print(f'1 paragraph gold size {len_1p_gold}')

    len_candidates_3P = len(candidates_3P['candidate'])
    print(f'3 paragraph candidates size {len_candidates_3P}')
    len_gold_3P = len(gold_3P['query'])
    print(f'3 paragraph gold size {len_gold_3P}')


def create_eval_data(dataset, available_cases):
    # gold_citations = {"anchor_citation": [], "gold_citation": []}
    apd = {"anchor": [], "positive": []}  # single para and positive only
    mapd = {"anchor": [], "positive": []}  # multiple paras and positive only
    apnd = {"anchor": [], "positive": [], "negative": []}  # multiple paras and positive and negative
    mapnd = {"anchor": [], "positive": [], "negative": []}  # multiple paras and positive and negative

    for data in tqdm(dataset):
        for paragraph in data['paragraph_citations']:
            citation = paragraph['citation']
            if citation['citation'].replace('\n', '').strip() in available_cases and len(
                    citation['paragraphs']) > 0:
                case = data['case'].replace('\n', '').strip()
                # apt, ppt, apmt, ppmt = get_anchor_positive_paragraphs(case, citation['para'], citation)
                try:
                    apt, ppt, npt, apmt, ppmt, npmt = get_anchor_positive_negative_paragraphs(case, paragraph['para'],
                                                                                              citation)
                    apd['anchor'].append(apt)
                    apd['positive'].append(ppt)
                    mapd['anchor'].append(apmt)
                    mapd['positive'].append(ppmt)

                    apnd['anchor'].append(apt)
                    apnd['positive'].append(ppt)
                    apnd['negative'].append(npt)
                    mapnd['anchor'].append(apmt)
                    mapnd['positive'].append(ppmt)
                    mapnd['negative'].append(npmt)
                except KeyError as ex:
                    continue
    save_to_file(apd, "data/data_splits/training/eval_positive_W1.tsv")
    save_to_file(mapd, "data/data_splits/training/eval_positive_W3.tsv")
    save_to_file(apnd, "data/data_splits/training/eval_positive_negative_W1.tsv")
    save_to_file(mapnd, "data/data_splits/training/eval_positive_negative_W3.tsv")

    len_apd = len(apd['anchor'])
    print(f'anchor positive eval set size {len_apd}')
    len_apnd = len(apnd['anchor'])
    print(f'anchor positive negative eval set size {len_apnd}')
    len_mapd = len(mapd['anchor'])
    print(f'multiple anchor positive eval set size {len_mapd}')
    len_mapnd = len(mapnd['anchor'])
    print(f'multiple anchor positive negative eval set size {len_mapnd}')

if __name__ == '__main__':
    with open("data/annotation/training.json", "r", encoding="utf-8") as f:
        training = json.load(f)

    with open("data/annotation/dev.json", "r", encoding="utf-8") as f:
        dev = json.load(f)

    with open("data/annotation/test.json", "r", encoding="utf-8") as f:
        test = json.load(f)

    mappings = load_mappings()

    # create_supervised_dataset(training, set(mappings.keys()))

    # create_candidates_and_queries(test, set(mappings.keys()))

    create_eval_data(dev, set(mappings.keys()))
