import csv
import json
import os.path

from loguru import logger
from tqdm import tqdm

from GPT_prompter import Prompter

if os.path.exists('data/ltsc_labeled.json'):
    with open('data/ltsc_labeled.json', 'r', encoding='utf8') as f:
        dataset = json.load(f)
else:
    with open('latvian-tweet-sentiment-corpus/tweet_corpus.json', 'r', encoding='utf8') as f:
        dataset = json.load(f)

dataset_by_ground_truth = {}
for d in dataset:
    if d['sentiment'] not in dataset_by_ground_truth:
        dataset_by_ground_truth[d['sentiment']] = []
    dataset_by_ground_truth[d['sentiment']].append(d)

min_sentiment_group = min([len(dataset_by_ground_truth[k]) for k in dataset_by_ground_truth])
for k in dataset_by_ground_truth:
    dataset_by_ground_truth[k] = dataset_by_ground_truth[k][:min_sentiment_group]

dataset = []
for k in dataset_by_ground_truth:
    dataset += dataset_by_ground_truth[k]


for i, d in enumerate(tqdm(dataset)):
    try:
        Prompter.label_sample(d)
    except Exception as e:
        logger.exception(e)
        logger.error(f'Failed to get response for {d["text"]}')
        with open('data/ltsc_labeled.json', 'w', encoding='utf8') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
        exit(0)
    if i % 10 == 0:
        with open('data/ltsc_labeled.json', 'w', encoding='utf8') as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)

prompt_metrics = Prompter.get_metrics(dataset)

# with open('data/ltsc_labeled_metrics.json', 'w') as f:
#     json.dump(prompt_metrics, f, indent=4, ensure_ascii=False)

with open('data/ltsc_labeled_metrics.csv', 'w', encoding='utf8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'prompt',
        'prompt_idx',
        'prompt_repeat_idx',
        'accuracy',
        'precision_avg',
        'recall_avg',
        'f1',
        'parsable_response_percent',
        'parsable_response_count',
        'precision_negative',
        'precision_neutral',
        'precision_positive',
        'recall_negative',
        'recall_neutral',
        'recall_positive',
        'prompt_text'
    ])
    for prompt in prompt_metrics:
        writer.writerow([
            prompt,
            prompt_metrics[prompt]['prompt_idx'],
            prompt_metrics[prompt]['prompt_repeat_idx'],
            prompt_metrics[prompt]['accuracy'],
            prompt_metrics[prompt]['precision_avg'],
            prompt_metrics[prompt]['recall_avg'],
            prompt_metrics[prompt]['f1'],
            prompt_metrics[prompt]['percent_parsable_responses'],
            prompt_metrics[prompt]['count_parsable_responses'],
            prompt_metrics[prompt]['precision']['negative'],
            prompt_metrics[prompt]['precision']['neutral'],
            prompt_metrics[prompt]['precision']['positive'],
            prompt_metrics[prompt]['recall']['negative'],
            prompt_metrics[prompt]['recall']['neutral'],
            prompt_metrics[prompt]['recall']['positive'],
            prompt_metrics[prompt]['prompt_text']
        ])

