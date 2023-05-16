import json
import random
import re
from collections import Counter
from multiprocessing.pool import ThreadPool

from loguru import logger
from tqdm import tqdm

from GPT_prompter import Prompter
from corpus_stats import flatten_comments_from_posts, get_stats

if __name__ == '__main__':
    with open('data/LVReddit_labeled.json', 'r', encoding='utf8') as f:
        dataset = json.load(f)

    samples_per_sentiment = {}
    for d in dataset:
        if d['sentiment'] not in samples_per_sentiment:
            samples_per_sentiment[d['sentiment']] = []
        samples_per_sentiment[d['sentiment']].append(d)

    for k in samples_per_sentiment:
        print(f'{k}: {len(samples_per_sentiment[k])}')
    print(sum([len(samples_per_sentiment[k]) for k in samples_per_sentiment]))


