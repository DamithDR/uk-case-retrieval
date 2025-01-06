import json
import os

from tqdm import tqdm


def run():
    files_path = 'data/raw/files'
    case_mapping = {}

    # List all files in the folder
    files = os.listdir(files_path)

    for file in tqdm(files):
        if file.endswith('.json'):
            with open(files_path + "/" + file, 'r',encoding='utf-8') as f:
                case = json.load(f)
                case_mapping[case['neutral_citation']] = file

    print(f'total mappings : {len(case_mapping.keys())}')
    with open('data/mapping/mapping.json', 'w', encoding='utf-8') as f:
        json.dump(case_mapping, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    run()
