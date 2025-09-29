import json

from rank_bm25 import BM25Okapi

from setting.retrieval_settings import setting
from stats.data_stats import load_json
from util.eval_utils import evaluate_results
from util.file_utils import write_results

candidates_file = 'data/annotation/candidates.json'
queries_file = 'data/annotation/queries.json'

citation_map = load_json('mapping.json', 'data/mapping/')
corpus = []


def generate_BM25_corpus():
    with open(candidates_file, 'r') as f:
        candidates = json.load(f)

        candidates_cases = [candidate['case'].replace('\n','').strip() for candidate in candidates]

        for candidate_case in candidates_cases:
            with open(f'data/raw/anonymised/{citation_map[candidate_case]}', 'r') as c_file:
                candidate_case_json = json.load(c_file)
                corpus.append("\n".join(candidate_case_json['full_text']))
    return candidates_cases


def get_scores(query):
    tokenized_query = query.lower().split()

    return bm25.get_scores(tokenized_query)


if __name__ == '__main__':
    candidates_cases = generate_BM25_corpus()
    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(queries_file, 'r') as f:
        queries = json.load(f)
        query_cases = [query['case'].replace('\n','').strip() for query in queries]

        results = {}
        for case in query_cases:
            with open(f'data/raw/anonymised/{citation_map[case]}', 'r') as c_file:
                case_json = json.load(c_file)
                query = "\n".join(case_json['full_text'])
                scores = get_scores(query)
                results[case] = {'keys': candidates_cases.copy(), 'scores': scores}

        MAP, f1, p, r = evaluate_results(results, queries, setting['k_values'])

        write_results(MAP, f, p, r, "BM25", setting['k_values'])
