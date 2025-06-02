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


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    files_path = 'data/raw/files'
    anon_path = 'data/raw/anonymised'

    # List all files in the folder
    files = os.listdir(files_path)

    for file in tqdm(files):
        if file.endswith('.json'):
            with open(files_path + "/" + file, 'r', encoding='utf-8') as f:
                original_case = json.load(f)
                anonymised_case = anonymize_names(original_case)
                with open(anon_path + "/" + file, "w", encoding="utf-8") as fr:
                    json.dump(anonymised_case, fr, ensure_ascii=False, indent=4)
