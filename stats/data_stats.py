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
def load_json(file_path, base_path='data/raw/files/'):
    with open(base_path + file_path, 'r') as f:
        return json.load(f)


def count_tokens(text):
    return len(text.split())


def parse_paragraphs(json_data, citation_map):
    """Extract paragraphs and token information from referenced files."""
    file_path = 'data/raw/files/'
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
    paragraphs = parse_paragraphs(data, citation_map)
    return len(paragraphs) / len(data) if data else 0


def get_avg_tokens_per_document(file_path, citation_map):
    data = load_json(file_path, 'data/annotation/')
    token_counts = []
    for obj in tqdm(data, desc="parsing documents"):
        document_tokens = 0
        for citation in obj.get("citations", []):
            if citation in citation_map:
                citation_file = citation_map[citation]
                if os.path.exists('data/raw/files/' + citation_file):
                    cited_data = load_json(citation_file)
                    for para_key in cited_data.get("sequence", []):
                        para_content = cited_data["paragraphs"].get(para_key, {}).get("paragraph", "")
                        document_tokens += count_tokens(para_content)
        token_counts.append(document_tokens)
    return calculate_average(token_counts)


def get_avg_tokens_per_paragraph(file_path, citation_map):
    data = load_json(file_path, 'data/annotation/')
    paragraphs = parse_paragraphs(data, citation_map)
    token_counts = [count_tokens(para) for para in paragraphs]
    return calculate_average(token_counts)


def plot_paragraph_token_frequencies_histogram(file_path, citation_map):
    data = load_json(file_path, 'data/annotation/')
    paragraphs = parse_paragraphs(data, citation_map)
    token_counts = [count_tokens(para) for para in paragraphs]
    plt.hist(token_counts, bins=20, edgecolor='black')
    plt.title('Paragraph Token Frequencies')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.show()


def plot_document_token_frequencies_histogram(file_path, citation_map):
    data = load_json(file_path, 'data/annotation/')
    token_counts = []
    for obj in data:
        document_tokens = 0
        for citation in obj.get("citations", []):
            if citation in citation_map:
                citation_file = citation_map[citation]
                if os.path.exists(citation_file):
                    cited_data = load_json(citation_file, 'data/mapping/')
                    for para_key in cited_data.get("sequence", []):
                        para_content = cited_data["paragraphs"].get(para_key, {}).get("paragraph", "")
                        document_tokens += count_tokens(para_content)
        token_counts.append(document_tokens)
    plt.hist(token_counts, bins=20, edgecolor='black')
    plt.title('Document Token Frequencies')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.show()


# Load the citation map
# citation_map_path = 'data/mapping/mapping.json'
citation_map = load_json('mapping.json', 'data/mapping/')

# Example usage
print("Average number of paragraphs (training):", get_avg_no_paragraphs(training, citation_map))
print("Average tokens per document (training):", get_avg_tokens_per_document(training, citation_map))
print("Average tokens per paragraph (training):", get_avg_tokens_per_paragraph(training, citation_map))

plot_paragraph_token_frequencies_histogram(training, citation_map)
plot_document_token_frequencies_histogram(training, citation_map)
