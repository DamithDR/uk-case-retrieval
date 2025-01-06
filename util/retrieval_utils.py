import json


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
    with open('data/raw/files/' + file_path, 'r') as file:
        return json.load(file)


def load_mappings():
    with open('data/mapping/mapping.json', 'r') as file:
        return json.load(file)
