import json

with open("data/annotation/training.json", "r", encoding="utf-8") as f:
    training = json.load(f)

with open("data/annotation/dev.json", "r", encoding="utf-8") as f:
    dev = json.load(f)

with open("data/annotation/test.json", "r", encoding="utf-8") as f:
    test = json.load(f)

training = [case_doc['case'] for case_doc in training]
dev = [case_doc['case'] for case_doc in dev]

test_citations = [citation for case_doc in test for citation in case_doc['citations']]
test = [case_doc['case'] for case_doc in test]


print(f'length : {len(training)}')
print(f'length : {len(test)}')
print(f'length : {len(dev)}')


expected_test_set = set(test_citations)
dev_set = set(dev)
train_set = set(training)

dev_inter = expected_test_set.intersection(dev_set)
train_inter = expected_test_set.intersection(train_set)
test_inter = expected_test_set.intersection(test)
dev_percentage = len(dev_inter) * 1.0 / len(expected_test_set) * 100
test_percentage = len(test_inter) * 1.0 / len(expected_test_set) * 100
training_percentage = len(train_inter) * 1.0 / len(expected_test_set) * 100
print(f'percentage of candidates in test : {test_percentage}')
print(f'percentage of candidates in dev : {dev_percentage}')
print(f'percentage of candidates in training : {training_percentage}')
