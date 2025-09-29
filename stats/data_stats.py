# training = 'data/annotation/training.json'
# dev = 'data/annotation/dev.json'
# test = 'data/annotation/test.json'


import json
from collections import Counter
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

# from tokenizers import Tokenizer

# Define the file paths
training = 'training.json'
dev = 'dev.json'
test = 'test.json'


# Load a BPE tokenizer
# tokenizer = Tokenizer.from_file('path_to_bpe_tokenizer.json')


# Helper functions
def load_json(file_path, base_path='data/raw/anonymised/'):
    with open(base_path + file_path, 'r') as f:
        return json.load(f)


def count_tokens(text):
    return len(text.split())


def parse_paragraphs(json_data, citation_map):
    """Extract paragraphs and token information from referenced files."""
    file_path = 'data/raw/anonymised/'
    paragraphs = []
    for obj in tqdm(json_data, desc='parsing paragraps'):
        for citation in obj.get("citations", []):
            if citation in citation_map:
                citation_file = citation_map[citation]
                if os.path.exists(file_path + citation_file):
                    cited_data = load_json(citation_file, file_path)
                    for para_key in cited_data.get("sequence", []):
                        para_content = cited_data["paragraphs"].get(para_key, {}).get("paragraph", "")
                        paragraphs.append(para_content)
    return paragraphs


def analyze_paragraphs(paragraphs):
    token_counts = [count_tokens(para) for para in paragraphs]
    return len(paragraphs), token_counts


def calculate_average(values):
    return sum(values) / len(values) if values else 0


def get_avg_no_paragraphs(file_path, citation_map):
    data = load_json(file_path, 'data/annotation/')
    data = [case['case'] for case in data]

    total_num_paras = []
    for case in data:
        case = case.replace('\n','').strip()
        case_doc = load_json(citation_map[case], 'data/raw/anonymised/')
        para_list = case_doc['full_text']
        total_num_paras.append(len(para_list))
    return sum(total_num_paras) * 1.0 / len(data)


def get_avg_tokens_per_document(file_path, citation_map):
    data = load_json(file_path, 'data/annotation/')
    data = [case['case'] for case in data]

    doc_wise_tokens = []
    for case in data:
        case = case.replace('\n', '').strip()
        case_doc = load_json(citation_map[case], 'data/raw/anonymised/')
        para_list = case_doc['full_text']
        tokens_in_doc = 0
        for para in para_list:
            tokens_in_doc += len(para.split(" "))  # using space tokeniser
        doc_wise_tokens.append(tokens_in_doc)
    return sum(doc_wise_tokens) * 1.0 / len(data)


def get_avg_tokens_per_paragraph(file_path, citation_map):
    data = load_json(file_path, 'data/annotation/')
    data = [case['case'] for case in data]

    doc_wise_tokens = []
    total_no_of_paras = 0
    for case in data:
        case = case.replace('\n', '').strip()
        case_doc = load_json(citation_map[case], 'data/raw/anonymised/')
        para_list = case_doc['full_text']
        tokens_in_doc = 0

        total_no_of_paras += len(para_list)
        for para in para_list:
            tokens_in_doc += len(para.split(" "))  # using space tokeniser
        doc_wise_tokens.append(tokens_in_doc)
    return sum(doc_wise_tokens) * 1.0 / total_no_of_paras


def plot_paragraph_token_frequencies_histogram(file_path, citation_map):
    data = load_json(file_path, 'data/annotation/')
    data = [case['case'] for case in data]

    para_wise_token_counts = []
    for case in data:
        case = case.replace('\n', '').strip()
        case_doc = load_json(citation_map[case], 'data/raw/anonymised/')
        para_list = case_doc['full_text']

        for para in para_list:
            para_wise_token_counts.append(len(para.split(" ")))  # using space tokeniser

    # paragraphs = parse_paragraphs(data, citation_map)
    # token_counts = [count_tokens(para) for para in paragraphs]
    plt.hist(para_wise_token_counts, bins=200, edgecolor='black')
    plt.title('Paragraph Token Frequencies')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.show()


def plot_document_token_frequencies_histogram(file_path, citation_map):
    data = load_json(file_path, 'data/annotation/')

    data = [case['case'] for case in data]

    doc_wise_tokens = []
    for case in data:
        case = case.replace('\n', '').strip()
        case_doc = load_json(citation_map[case], 'data/raw/anonymised/')
        para_list = case_doc['full_text']
        tokens_in_doc = 0
        for para in para_list:
            tokens_in_doc += len(para.split(" "))  # using space tokeniser
        doc_wise_tokens.append(tokens_in_doc)


    plt.hist(doc_wise_tokens, bins=200, edgecolor='black')
    plt.title('Document Token Frequencies')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == '__main__':


    # Load the citation map
    # citation_map_path = 'data/mapping/mapping.json'
    citation_map = load_json('mapping.json', 'data/mapping/')


    # Example usage
    # print("Average number of paragraphs (dev):", get_avg_no_paragraphs(dev, citation_map))
    # print("Average tokens per document (dev):", get_avg_tokens_per_document(dev, citation_map))
    # print("Average tokens per paragraph (dev):", get_avg_tokens_per_paragraph(dev, citation_map))


    plot_paragraph_token_frequencies_histogram(dev, citation_map)
    plot_document_token_frequencies_histogram(dev, citation_map)
