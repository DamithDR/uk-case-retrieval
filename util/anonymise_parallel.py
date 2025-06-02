import json
import os
import spacy
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# GLOBAL variable for each process
nlp = None

def init_process():
    global nlp
    nlp = spacy.load("en_core_web_sm")

def anonymize_file(file_info):
    global nlp
    file_path, output_path = file_info

    with open(file_path, 'r', encoding='utf-8') as f:
        case = json.load(f)

    sequence = case['sequence']
    anon_full_text = []

    for number in sequence:
        original_text = case['paragraphs'][number]['paragraph']
        doc = nlp(original_text)

        anonymized_text = original_text
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                anonymized_text = anonymized_text.replace(ent.text, "[PERSON]")
        case['paragraphs'][number]['paragraph'] = anonymized_text
        anon_full_text.append(anonymized_text)

    case['full_text'] = anon_full_text

    with open(output_path, "w", encoding="utf-8") as fr:
        json.dump(case, fr, ensure_ascii=False, indent=4)

    return output_path

if __name__ == '__main__':
    files_path = 'data/raw/files'
    anon_path = 'data/raw/anonymised'
    os.makedirs(anon_path, exist_ok=True)

    file_list = [
        (os.path.join(files_path, f), os.path.join(anon_path, f))
        for f in os.listdir(files_path) if f.endswith('.json')
    ]

    with Pool(processes=4, initializer=init_process) as pool:
        for _ in tqdm(pool.imap_unordered(anonymize_file, file_list), total=len(file_list)):
            pass
