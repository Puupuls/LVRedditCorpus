import json

import torch
from langdetect import detect_langs
from loguru import logger
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from transformers import pipeline

file = 'data/reddit_data_latvia_16753_2023-03-30T01:30:38.975318.json'

with open('SentimentWordsLV/positive.txt', 'r', encoding='utf8') as f:
    for i in range(10):
        f.readline()
    SentimentWordsLV_positive = set([word.strip() for word in f.readlines()])
with open('SentimentWordsLV/negative.txt', 'r', encoding='utf8') as f:
    for i in range(10):
        f.readline()
    SentimentWordsLV_negative = set([word.strip() for word in f.readlines()])

with open('om/updates/poz.final.txt', 'r', encoding='utf8') as f:
    for i in range(1):
        f.readline()
    om_positive = set([word.strip() for word in f.readlines()])
with open('om/updates/neg.final.txt', 'r', encoding='utf8') as f:
    for i in range(1):
        f.readline()
    om_negative = set([word.strip() for word in f.readlines()])
with open('om/updates/stopwords.txt', 'r', encoding='utf8') as f:
    # List of words that do not hold sentiment....
    # And some that do, like "Paldies", "😊", "👍" at the end
    for i in range(2):
        f.readline()
    om_stopwords = set([word.strip() for word in f.readlines()][:-7])

sia = SentimentIntensityAnalyzer()  # ~64% acc, uses https://github.com/cjhutto/vaderSentiment

MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
xml_ro_pipeline = pipeline(
    "sentiment-analysis",
    model=MODEL,
    tokenizer=MODEL,
    max_length=512,
    truncation=True,
    top_k=None,
    device=0 if DEVICE == 'cuda' else -1
)


def tag_recursive(post):
    body = post['body']
    post['lang_confidences'] = []
    post['lang'] = 'unknown'
    try:
        post['lang_confidences'] = detect_langs(body)
        post['lang'] = post['lang_confidences'][0].lang if post['lang_confidences'] else 'unknown'
        post['lang_confidences'] = {lang.lang: lang.prob for lang in post['lang_confidences']}
    except Exception as e:
        pass

    body_split = body.lower()
    body_split = body_split.replace('\n', ' ')
    body_split = body_split.replace('\r', ' ')
    body_split = body_split.replace('\t', ' ')
    body_split = body_split.replace('.', ' ')
    body_split = body_split.replace(',', ' ')
    body_split = body_split.replace('!', ' ')
    body_split = body_split.replace('?', ' ')
    body_split = body_split.split(' ')
    body_split = [word for word in body_split if not word.startswith('http')]
    words_resplit = []
    for word in body_split:
        if '/' in word:
            # Solve cases whn users write options separated by / without spaces
            words_resplit += word.split('/')
        else:
            words_resplit.append(word)
    body_split = words_resplit
    body_split = [word for word in body_split if word]
    body_split = [word for word in body_split if word not in om_stopwords]

    post['sentiment_detailed'] = {}
    post['sentiment_detailed']['SentimentWordsLV_positive'] = 0
    post['sentiment_detailed']['SentimentWordsLV_negative'] = 0
    post['sentiment_detailed']['om_positive'] = 0
    post['sentiment_detailed']['om_negative'] = 0
    post['sentiment_detailed']['xml_roberta_positive'] = 0
    post['sentiment_detailed']['xml_roberta_negative'] = 0
    post['sentiment_detailed']['xml_roberta_neutral'] = 0
    try:
        body_split = [word for word in body_split if word]
        if post['lang'] == 'lv':
            post['sentiment_detailed']['SentimentWordsLV_positive'] = sum(
                [1 for word in body_split if word in SentimentWordsLV_positive])
            post['sentiment_detailed']['SentimentWordsLV_negative'] = sum(
                [1 for word in body_split if word in SentimentWordsLV_negative])
            post['sentiment_detailed']['om_positive'] = sum([1 for word in body_split if word in om_positive])
            post['sentiment_detailed']['om_negative'] = sum([1 for word in body_split if word in om_negative])

        output = xml_ro_pipeline(body)
        for sentiment in output[0]:
            post['sentiment_detailed'][f'xml_roberta_{sentiment["label"]}'] += sentiment['score']
    except Exception as e:
        logger.exception(e)
        pass

    sentiment = (
            (
                    post['sentiment_detailed']['xml_roberta_positive'] - post['sentiment_detailed']['xml_roberta_negative']
            ) * 2
            # + (
            #         post['sentiment_detailed']['SentimentWordsLV_positive'] - post['sentiment_detailed']['SentimentWordsLV_negative'] +
            #         post['sentiment_detailed']['om_positive'] - post['sentiment_detailed']['om_negative']
            # )
    )

    if sentiment > 1:
        post['sentiment'] = 'positive'
    elif sentiment < -1:
        post['sentiment'] = 'negative'
    else:
        post['sentiment'] = 'neutral'

    if 'replies' in post:
        for reply in post['replies']:
            tag_recursive(reply)


if __name__ == '__main__':
    with open(file, 'r') as f:
        data = json.load(f)

    for post in tqdm(data):
        tag_recursive(post)

    with open(file, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
