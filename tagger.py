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
    SentimentWordsLV_positive = set([word.strip() for word in f.readlines()])
with open('SentimentWordsLV/negative.txt', 'r') as f:
    for i in range(10):
        f.readline()
    SentimentWordsLV_negative = set([word.strip() for word in f.readlines()])

with open('om/updates/poz.final.txt', 'r') as f:
    for i in range(1):
        f.readline()
    om_positive = set([word.strip() for word in f.readlines()])
with open('om/updates/neg.final.txt', 'r') as f:
    for i in range(1):
        f.readline()
    om_negative = set([word.strip() for word in f.readlines()])
with open('om/updates/stopwords.txt', 'r') as f:
    for i in range(2):
        f.readline()
    om_stopwords = set([word.strip() for word in f.readlines()])


sia = SentimentIntensityAnalyzer()  # ~64% acc, uses https://github.com/cjhutto/vaderSentiment

MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
xml_ro_tokenizer = AutoTokenizer.from_pretrained(MODEL)
xml_ro_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
xml_ro_model.to(DEVICE)
xml_ro_config = AutoConfig.from_pretrained(MODEL)


def tag_recursive(post):
    body = post['body'] if 'body' in post else f"{post['title']}\n{post['selftext'] if '[deleted]' != post['selftext'] else post['original_selftext']}"
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
    body_split = [word for word in body_split if word not in om_stopwords]

    post['sentiment_detailed'] = {}
    post['sentiment_detailed']['SentimentWordsLV_positive'] = 0
    post['sentiment_detailed']['SentimentWordsLV_negative'] = 0
    post['sentiment_detailed']['om_positive'] = 0
    post['sentiment_detailed']['om_negative'] = 0
    post['sentiment_detailed']['nltk_positive'] = 0
    post['sentiment_detailed']['nltk_negative'] = 0
    post['sentiment_detailed']['nltk_neutral'] = 0
    post['sentiment_detailed']['nltk_compound'] = 0
    post['sentiment_detailed']['xml_roberta_positive'] = 0
    post['sentiment_detailed']['xml_roberta_negative'] = 0
    post['sentiment_detailed']['xml_roberta_neutral'] = 0
    try:
        body_split = [word for word in body_split if word]
        if post['lang'] == 'lv':
            post['sentiment_detailed']['SentimentWordsLV_positive'] = sum([1 for word in body_split if word in SentimentWordsLV_positive])
            post['sentiment_detailed']['SentimentWordsLV_negative'] = sum([1 for word in body_split if word in SentimentWordsLV_negative])
            post['sentiment_detailed']['om_positive'] = sum([1 for word in body_split if word in om_positive])
            post['sentiment_detailed']['om_negative'] = sum([1 for word in body_split if word in om_negative])
        if post['lang'] == 'en':
            post['sentiment_detailed']['nltk_positive'] = sia.polarity_scores(body)['pos']
            post['sentiment_detailed']['nltk_negative'] = sia.polarity_scores(body)['neg']
            post['sentiment_detailed']['nltk_neutral'] = sia.polarity_scores(body)['neu']
            post['sentiment_detailed']['nltk_compound'] = sia.polarity_scores(body)['compound']

        encoded_input = xml_ro_tokenizer(body, return_tensors='pt').to(DEVICE)
        output = xml_ro_model(**encoded_input)
        scores = output[0][0].cpu().detach().numpy()
        scores = softmax(scores)
        post['sentiment_detailed']['xml_roberta_negative'] = float(scores[0])
        post['sentiment_detailed']['xml_roberta_neutral'] = float(scores[1])
        post['sentiment_detailed']['xml_roberta_positive'] = float(scores[2])
    except:
        pass

    sentiment = (
                        post['sentiment_detailed']['SentimentWordsLV_positive'] - post['sentiment_detailed']['SentimentWordsLV_negative'] +
                        post['sentiment_detailed']['om_positive'] - post['sentiment_detailed']['om_negative']
                ) / 2 + \
                post['sentiment_detailed']['nltk_compound']*2 + \
                (
                        post['sentiment_detailed']['xml_roberta_positive'] - post['sentiment_detailed']['xml_roberta_negative']
                ) * 2

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
