import json
import os
import spacy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to be called in each thread
def anonymize_names_thread(file_path, output_path):
    nlp = spacy.load("en_core_web_sm")  # Create separate nlp instance per thread

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

    # Save output
    with open(output_path, "w", encoding="utf-8") as fr:
        json.dump(case, fr, ensure_ascii=False, indent=4)

    return file_path


if __name__ == '__main__':
    files_path = 'data/raw/files'
    anon_path = 'data/raw/anonymised'

    os.makedirs(anon_path, exist_ok=True)

    files = [f for f in os.listdir(files_path) if f.endswith('.json')]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file in files:
            file_path = os.path.join(files_path, file)
            output_path = os.path.join(anon_path, file)
            futures.append(executor.submit(anonymize_names_thread, file_path, output_path))

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # Optional: handles exceptions
