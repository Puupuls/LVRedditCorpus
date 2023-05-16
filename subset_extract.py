import json
import random
import re


if __name__ == '__main__':
    with open('data/LVReddit_labeled.json', 'r', encoding='utf8') as f:
        dataset = json.load(f)

    num_positive = len([d for d in dataset if d['sentiment'] == 'positive'])
    num_negative = len([d for d in dataset if d['sentiment'] == 'negative'])
    num_neutral = len([d for d in dataset if d['sentiment'] == 'neutral'])
    print(f'Positive: {num_positive}')
    print(f'Negative: {num_negative}')
    print(f'Neutral: {num_neutral}')

    dataset_by_sentiment = {}
    for d in dataset:
        if d['sentiment'] not in dataset_by_sentiment:
            dataset_by_sentiment[d['sentiment']] = []
        dataset_by_sentiment[d['sentiment']].append(d)

    # select 100 from each
    for k in dataset_by_sentiment:
        dataset_by_sentiment[k] = random.sample(dataset_by_sentiment[k], 100)

    ds = []
    for k in dataset_by_sentiment:
        ds.extend(dataset_by_sentiment[k])

    with open('data/subset_labeled.json', 'w', encoding='utf8') as f:
        json.dump(ds, f, indent=4, ensure_ascii=False)
