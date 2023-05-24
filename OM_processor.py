import json

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


def getSentiment(text):
    cleaned_text = [word for word in text.lower().split() if word not in om_stopwords]
    positive = len([word for word in cleaned_text if word in om_positive])
    negative = len([word for word in cleaned_text if word in om_negative])
    if positive - negative > 0:
        return 'positive'
    elif negative - positive > 0:
        return 'negative'
    else:
        return 'neutral'



with open('latvian-tweet-sentiment-corpus/tweet_corpus.json', 'r', encoding='utf8') as f:
    dataset = json.load(f)
correct = 0
total = 0
for d in dataset:
    sentiment = {"POZ": 'positive', "NEG": 'negative', "NEU": 'neutral'}[d['sentiment']]
    om_sentiment = getSentiment(d['text'])
    if sentiment == om_sentiment:
        correct += 1
    total += 1
print(f'latvian-tweet-sentiment-corpus accuracy: {correct / total}')


with open('LV-twitter-sentiment-corpus/tweet_corpus.json', 'r', encoding='utf8') as f:
    dataset = json.load(f)
correct = 0
total = 0
for d in dataset:
    sentiment = {"POZ": 'positive', "NEG": 'negative', "NEU": 'neutral'}[d['sentiment']]
    om_sentiment = getSentiment(d['text'])
    if sentiment == om_sentiment:
        correct += 1
    total += 1
print(f'LV-twitter-sentiment-corpus accuracy: {correct / total}')


with open('Latvian-Twitter-Eater-Corpus/sub-corpora/sentiment-analysis/ltec-sentiment-annotated-test.json', 'r', encoding='utf8') as f:
    dataset = json.load(f)
correct = 0
total = 0
for d in dataset:
    sentiment = {"pos": 'positive', "neg": 'negative', "neu": 'neutral'}[d['sentiment']]
    om_sentiment = getSentiment(d['tweet_text'])
    if sentiment == om_sentiment:
        correct += 1
    total += 1
print(f'Latvian Twitter Eater Corpus accuracy: {correct / total}')


with open('sikzinu_analize/viksna.json', 'r', encoding='utf8') as f:
    dataset = json.load(f)
correct = 0
total = 0
for d in dataset['data']:
    sentiment = 'positive' if d['POS'] - d['NEG'] > 0 else 'negative' if d['NEG'] - d['POS'] > 0 else 'neutral'
    om_sentiment = getSentiment(d['text'])
    if not d['NOT_LV']:
        if sentiment == om_sentiment:
            correct += 1
        total += 1
print(f'sikzinu_analize accuracy: {correct / total}')