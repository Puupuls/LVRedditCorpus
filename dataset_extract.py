import json
import random
import re

from stats import flatten_comments_from_posts, get_stats

file = 'data/reddit_data_latvia_16753_2023-03-30T01:30:38.975318.json'


def extract_dataset(data):
    dataset = []
    for post in data + flatten_comments_from_posts(data):
        if post['lang'] == 'lv' and post['body'] and 'http' not in post['body']:
            body_cleaned = post['body'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            body_cleaned = re.sub(r'\s+', ' ', body_cleaned)
            body_cleaned = body_cleaned.strip()
            body_len = len(body_cleaned)
            if 30 < body_len < 1000:
                post = {
                    'id': post['name'],
                    'body': body_cleaned,
                    'sentiment': post['sentiment'],
                    'lang': post['lang'],
                    'sentiment_detailed': post['sentiment_detailed'],
                    'lang_confidences': post['lang_confidences'],
                }
                dataset.append(post)
    dataset_by_sentiment = {
        'positive': [],
        'neutral': [],
        'negative': [],
    }
    for post in dataset:
        dataset_by_sentiment[post['sentiment']].append(post)

    dataset_by_sentiment_small = {
        'positive': [],
        'neutral': [],
        'negative': [],
    }
    for sentiment, posts in dataset_by_sentiment.items():
        random.shuffle(posts)
        dataset_by_sentiment_small[sentiment] = posts[:150]

    return dataset_by_sentiment_small


if __name__ == '__main__':
    with open(file, 'r') as f:
        data = json.load(f)
    dataset = extract_dataset(data)
    print(f"Dataset size: {len(dataset)}")
    with open('data/dataset.json', 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    # get_stats(dataset, print_to_console=True)
