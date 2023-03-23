import json

import torch
from langdetect import detect_langs
from tqdm import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax

file = 'data/reddit_data_latvia_8169.json'

with open('SentimentWordsLV/positive.txt', 'r') as f:
    for i in range(10):
        f.readline()
    positive_words_lv = [word.strip() for word in f.readlines()]
with open('SentimentWordsLV/negative.txt', 'r') as f:
    for i in range(10):
        f.readline()
    negative_words_lv = [word.strip() for word in f.readlines()]

sia = SentimentIntensityAnalyzer()  # ~64% acc, uses https://github.com/cjhutto/vaderSentiment

MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
xml_ro_tokenizer = AutoTokenizer.from_pretrained(MODEL)
xml_ro_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
xml_ro_model.to(DEVICE)
xml_ro_config = AutoConfig.from_pretrained(MODEL)


def tag_recursive(post):
    body = post['body'] if 'body' in post else f"{post['title']}\n{post['selftext']}"
    try:
        post['lang_confidences'] = detect_langs(body)
        post['lang'] = post['lang_confidences'][0].lang if post['lang_confidences'] else 'unknown'
        post['lang_confidences'] = {lang.lang: lang.prob for lang in post['lang_confidences']}
    except Exception as e:
        post['lang_confidences'] = []
        post['lang'] = 'unknown'

    body_split = body.lower()
    body_split = body_split.replace('\n', ' ')
    body_split = body_split.replace('\r', ' ')
    body_split = body_split.replace('\t', ' ')
    body_split = body_split.replace('.', ' ')
    body_split = body_split.replace(',', ' ')
    body_split = body_split.replace('!', ' ')
    body_split = body_split.replace('?', ' ')
    body_split = body_split.replace('/', ' ')
    body_split = body_split.split(' ')

    post['SentimentWordsLV_positive'] = 0
    post['SentimentWordsLV_negative'] = 0
    post['nltk_positive'] = 0
    post['nltk_negative'] = 0
    post['nltk_neutral'] = 0
    post['nltk_compound'] = 0
    post['xml_roberta_positive'] = 0
    post['xml_roberta_negative'] = 0
    post['xml_roberta_neutral'] = 0
    try:
        body_split = [word for word in body_split if word]
        if post['lang'] == 'lv':
            post['SentimentWordsLV_positive'] = sum([1 for word in body_split if word in positive_words_lv])
            post['SentimentWordsLV_negative'] = sum([1 for word in body_split if word in negative_words_lv])
        if post['lang'] == 'en':
            post['nltk_positive'] = sia.polarity_scores(body)['pos']
            post['nltk_negative'] = sia.polarity_scores(body)['neg']
            post['nltk_neutral'] = sia.polarity_scores(body)['neu']
            post['nltk_compound'] = sia.polarity_scores(body)['compound']

        encoded_input = xml_ro_tokenizer(body, return_tensors='pt').to(DEVICE)
        output = xml_ro_model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        post['xml_roberta_negative'] = scores[0]
        post['xml_roberta_neutral'] = scores[1]
        post['xml_roberta_positive'] = scores[2]
    except:
        pass

    sentiment = post['SentimentWordsLV_positive'] - post['SentimentWordsLV_negative'] + \
                post['nltk_compound']*2 + \
                (post['xml_roberta_positive'] - post['xml_roberta_negative']) * 2

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
