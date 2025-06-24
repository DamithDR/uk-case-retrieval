import json

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

from util.retrieval_utils import load_mappings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name):
    t = AutoTokenizer.from_pretrained(model_name)
    m = AutoModel.from_pretrained(model_name)

    m = m.to(device)
    return m, t


def generate_embedding(paragraph):
    # Tokenize single paragraph
    inputs = tokenizer(
        paragraph,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Move inputs to device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Compute embeddings without gradient
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling (excluding padding)
    attention_mask = inputs["attention_mask"]
    hidden_states = outputs.last_hidden_state  # Shape: (1, seq_len, hidden_dim)
    masked_sum = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
    token_counts = attention_mask.sum(dim=1, keepdim=True)
    mean_embedding = masked_sum / token_counts.clamp(min=1)

    return mean_embedding.squeeze().cpu().numpy()  # Shape: (hidden_dim,)


def encode_long_paragraph(paragraph, max_length=512, stride=256):
    tokens = tokenizer(paragraph, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    chunk_embeddings = []

    for start in range(0, len(tokens), stride):
        end = min(start + max_length - 2, len(tokens))  # Reserve space for [CLS], [SEP]
        chunk = tokens[start:end]
        inputs = tokenizer.decode(chunk, skip_special_tokens=True)
        inputs = tokenizer(inputs, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        mean_embedding = (outputs.last_hidden_state * inputs["attention_mask"].unsqueeze(-1)).sum(dim=1)
        token_counts = inputs["attention_mask"].sum(dim=1, keepdim=True)
        mean_embedding = (mean_embedding / token_counts.clamp(min=1)).cpu().numpy()
        chunk_embeddings.append(mean_embedding)

    return np.mean(chunk_embeddings, axis=0)


def load_file_and_gen(file_name):
    file_path = f'data/raw/anonymised/{file_name}'
    with open(file_path, 'r') as f:
        case_to_embedd = json.load(f)
        paragraph_keys = case_to_embedd['paragraphs'].keys()
        file_wise_embedding = [encode_long_paragraph(case_to_embedd['paragraphs'][p]['paragraph']) if len(
            tokenizer.encode(case_to_embedd['paragraphs'][p]['paragraph'])) > 512 else generate_embedding(
            case_to_embedd['paragraphs'][p]['paragraph']) for p in
                               tqdm(paragraph_keys)]
        np.save(f"embeddings/{file_name}.npy", file_wise_embedding)


if __name__ == '__main__':
    model_name = 'nlpaueb/legal-bert-base-uncased'  # parameterise
    model, tokenizer = load_model(model_name)
    mapping = load_mappings()

    dev_data = 'data/annotation/dev.json'

    with open(dev_data, 'r') as file:
        annotation = json.load(file)
        for document in tqdm(annotation):
            source_citation = document['case']
            cited_documents = document['citations']

            source_file = mapping[source_citation]
            load_file_and_gen(source_file)

            for cited_file in [mapping[cited] for cited in cited_documents]:
                load_file_and_gen(cited_file)
