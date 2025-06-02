import json
import os

import spacy
from tqdm import tqdm


def anonymize_names(case):
    sequence = case['sequence']

    # anon the paragraphs
    anon_full_text = []
    for number in sequence:

        original_text = case['paragraphs'][number]['paragraph']

        doc = nlp(original_text)

        # Replace person names with [PERSON]
        anonymized_text = original_text
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                anonymized_text = anonymized_text.replace(ent.text, "[PERSON]")
        case['paragraphs'][number]['paragraph'] = anonymized_text
        anon_full_text.append(anonymized_text)
    case['full_text'] = anon_full_text

    return case

def get_batch_indices(batch_number, total_batches, total_files):
    files_per_batch = total_files // total_batches
    remainder = total_files % total_batches  # leftover files

    # First `remainder` batches get one extra file
    if batch_number < remainder:
        start_index = batch_number * (files_per_batch + 1)
        end_index = start_index + files_per_batch + 1
    else:
        start_index = remainder * (files_per_batch + 1) + (batch_number - remainder) * files_per_batch
        end_index = start_index + files_per_batch

    return start_index, end_index


if __name__ == '__main__':
    array_id = os.environ.get('SLURM_ARRAY_TASK_ID', None)
    files_path = 'data/raw/files'
    anon_path = 'data/raw/anonymised'
    total_batches = 16

    if array_id is None:
        print("This script should be run as a SLURM array job.")
        exit(1)

    array_index = int(array_id)
    print(f"Running SLURM array job with ID: {array_id}")

    # List all files in the folder
    files = os.listdir(files_path)
    files.sort()

    total_files = len(files)

    start, end = get_batch_indices(array_index, total_batches, total_files)
    files = files[start:end]

    nlp = spacy.load("en_core_web_sm")

    for file in tqdm(files):
        if file.endswith('.json'):
            with open(files_path + "/" + file, 'r', encoding='utf-8') as f:
                original_case = json.load(f)
                anonymised_case = anonymize_names(original_case)
                with open(anon_path + "/" + file, "w", encoding="utf-8") as fr:
                    json.dump(anonymised_case, fr, ensure_ascii=False, indent=4)




