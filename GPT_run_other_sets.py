import json
from time import sleep

import openai
from loguru import logger
from tqdm import tqdm

from GPT_prompter import Prompter


def getSentiment(text):
    while True:
        try:
            try:
                resp = Prompter.get_response_chat(Prompter.prompts[15], text)
                break
            except openai.error.RateLimitError:
                logger.warning('Rate limit error, sleeping for 10 seconds')
                sleep(10)
                continue
        except Exception as e:
            logger.exception(e)
            logger.error(f'Failed to get response for {text}')

    resp_clean = resp.split("\n")[-1].lower()
    is_pos = False
    is_neg = False
    is_neu = False
    if 'negative' in resp_clean:
        is_neg = True
    if 'neutral' in resp_clean:
        is_neu = True
    if 'positive' in resp_clean:
        is_pos = True
    if is_pos and not is_neg and not is_neu:
        resp_clean = 'positive'
    elif is_neg and not is_pos and not is_neu:
        resp_clean = 'negative'
    elif is_neu and not is_pos and not is_neg:
        resp_clean = 'neutral'
    else:
        logger.info('Invalid response:', resp)
        resp_clean = 'neutral'
    if resp_clean not in ['negative', 'neutral', 'positive']:
        logger.info('Invalid response:', resp)
        resp_clean = 'neutral'
    return resp_clean


with open('LV-twitter-sentiment-corpus/tweet_corpus.json', 'r', encoding='utf8') as f:
    dataset = json.load(f)
correct = 0
total = 0
for d in tqdm(dataset):
    sentiment = {"POZ": 'positive', "NEG": 'negative', "NEU": 'neutral'}[d['sentiment']]
    gpt_sentiment = getSentiment(d['text'])
    if sentiment == gpt_sentiment:
        correct += 1
    total += 1
print(f'LV-twitter-sentiment-corpus accuracy: {correct / total}')


with open('Latvian-Twitter-Eater-Corpus/sub-corpora/sentiment-analysis/ltec-sentiment-annotated-test.json', 'r', encoding='utf8') as f:
    dataset = json.load(f)
correct = 0
total = 0
for d in tqdm(dataset):
    sentiment = {"pos": 'positive', "neg": 'negative', "neu": 'neutral'}[d['sentiment']]
    gpt_sentiment = getSentiment(d['tweet_text'])
    if sentiment == gpt_sentiment:
        correct += 1
    total += 1
print(f'Latvian Twitter Eater Corpus accuracy: {correct / total}')


with open('sikzinu_analize/viksna.json', 'r', encoding='utf8') as f:
    dataset = json.load(f)
correct = 0
total = 0
for i, d in tqdm(list(enumerate(dataset['data']))):
    sentiment = 'positive' if d['POS'] - d['NEG'] > 0 else 'negative' if d['NEG'] - d['POS'] > 0 else 'neutral'
    gpt_sentiment = getSentiment(d['text'])
    if not d['NOT_LV']:
        if sentiment == gpt_sentiment:
            correct += 1
        total += 1
    if i % 100 == 0:
        print(f'sikzinu_analize accuracy: {correct / total}')
        with open('sikzinu_analize/viksna_results.json', 'w', encoding='utf8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
print(f'sikzinu_analize accuracy: {correct / total}')



# with open('om/data/psgs_norm.arff', 'r', encoding='utf8') as f:
#     for i in range(8):
#         f.readline()
#     dataset = [[i.strip(' \'\n') for i in j.split(',')] for j in f.readlines()]
with open('om/data/psgs_norm.json', 'r', encoding='utf8') as f:
    dataset = json.load(f)
correct = 0
total = 0
for i, d in tqdm(list(enumerate(dataset))):
    sentiment = {"POZ": 'positive', "NEG": 'negative', "NEU": 'neutral'}[d[2]]
    if len(d) == 3:
        gpt_sentiment = getSentiment(d[1])
        d.append(gpt_sentiment)
    else:
        gpt_sentiment = d[3]
    if sentiment == gpt_sentiment:
        correct += 1
    total += 1
    if i % 100 == 0:
        print(f'om accuracy: {correct / total}')
        with open('om/data/psgs_norm.json', 'w', encoding='utf8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
print(f'OM accuracy: {correct / total}')