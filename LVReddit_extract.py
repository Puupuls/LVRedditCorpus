import json
import random
import re
from collections import Counter

from corpus_stats import flatten_comments_from_posts, get_stats

file = 'data/reddit_data_latvia_24237.json'


def extract_dataset(data):
    dataset = []
    for post in data + flatten_comments_from_posts(data):
        if post['lang'] == 'lv' and post['body'] and 'http' not in post['body']:
            body_cleaned = post['body'].replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            body_cleaned = re.sub(r'\s+', ' ', body_cleaned)
            body_cleaned = body_cleaned.strip()
            body_len = len(body_cleaned)
            if 30 < body_len < 1000:
                om_sentiment = post['sentiment_detailed']['om_positive'] - post['sentiment_detailed']['om_negative']
                SentimentWordsLV_sentiment = post['sentiment_detailed']['SentimentWordsLV_positive'] - post['sentiment_detailed']['SentimentWordsLV_negative']
                post = {
                    'id': post['name'],
                    'parent_id': post.get('parent_id'),
                    'type': 'comment' if post.get('parent_id') else 'post',
                    'depth': post.get('depth'),
                    'body': body_cleaned,
                    'lang': post['lang'],
                    'permalink': post['permalink'],
                    'labels': {
                        'twitter-xlm-roberta-base-sentiment': 'positive' if post['sentiment_detailed']['xml_roberta_positive'] > 0.5 else 'negative' if post['sentiment_detailed']['xml_roberta_negative'] > 0.5 else 'neutral',
                        'om': 'positive' if om_sentiment > 1 else 'negative' if om_sentiment < -1 else 'neutral',
                        'SentimentWordsLV': 'positive' if SentimentWordsLV_sentiment > 1 else 'negative' if SentimentWordsLV_sentiment < -1 else 'neutral',
                    }
                }
                dataset.append(post)

    return dataset


if __name__ == '__main__':
    with open(file, 'r', encoding='utf8') as f:
        data = json.load(f)
    dataset = extract_dataset(data)
    print(f"Dataset size: {len(dataset)}")
    # with open('data/LVReddit_dataset.json', 'w', encoding='utf8') as f:
    #     json.dump(dataset, f, indent=4, ensure_ascii=False)

    posts = [post for post in dataset if post['type'] == 'post']
    comments = [post for post in dataset if post['type'] == 'comment']
    print(f"Posts: {len(posts)}")
    print(f"Comments: {len(comments)}")
    for post in dataset:
        # Replace links with #link
        post['body'] = re.sub("\[.+\]\(https?://.*\) ?", '#link', post['body'])
        post['body'] = re.sub("https?://.* ?", '#link', post['body'])

    chars = Counter(
        [char for post in dataset for char in post['body']]
    )
    print(f"Most common characters: {chars.most_common(10)}")
    print(f"Least common characters: {chars.most_common()[:-11:-1]}")
    print(f"Character count: {sum(chars.values())} (unique: {len(chars)})")

    lower_chars = Counter(
        [char for post in dataset for char in post['body'].lower()]
    )
    print(f"Most common lower characters: {lower_chars.most_common(10)}")
    print(f"Least common lower characters: {lower_chars.most_common()[:-11:-1]}")
    print(f"Lower character count: {sum(lower_chars.values())} (unique: {len(lower_chars)})")
    lv_chars = 'aābcčdeēfgģhiījkķlļmnņoprsštuūvzž1234567890.,?!/:«»\\-\'\"*()[]{}+~;&#%_`@$ \n\t'
    print(f"Characters not in LV alphabet: {len([char for char in chars if char.lower() not in lv_chars])} ({len([char for char in chars if char.lower() not in lv_chars]) / len(lower_chars) * 100:.2f}%)")
    lv_chars = set(lv_chars)
    non_lv_count = 0
    for c in [char for post in dataset for char in post['body'].lower()]:
        if c not in lv_chars:
            non_lv_count += 1
    print(f"Characters not in LV alphabet: {non_lv_count} ({non_lv_count / sum(lower_chars.values()) * 100:.2f}%)")

    words = Counter(
        [word for post in dataset for word in re.sub(r'[.,?!/:«»\-\'\"\*\(\)\[\]\{\}\+\~]', ' ', post['body']).lower().split() if word]
    )
    print(f"Most common words: {words.most_common(10)}")
    print(f"Least common words: {words.most_common()[:-11:-1]}")
    print(f"Word count: {sum(words.values())} (unique: {len(words)})")
    with open('data/lv_dict_v2.txt', 'r', encoding='utf8') as f:
        lv_dict = f.read().splitlines()
        lv_dict = set(lv_dict)

    print(f"Words in LV dictionary: {len([word for word in words if word in lv_dict])} ({len([word for word in words if word in lv_dict]) / len(words) * 100:.2f}%)")
    actual_words = [word for word in words if re.match(r'^[a-zāčēģīķļņšūž]+$', word)]
    print(f"Alphanum words: {len(actual_words)} ({len(actual_words) / len(words) * 100:.2f}%)")

    word_lengths = [len(word) for word in words]
    word_lengths_c = Counter(
        [len(word) for word in words]
    )
    print(f"Min word length: {min(word_lengths)} (count: {word_lengths_c[min(word_lengths)]})")
    print(f"Q1: {sorted(word_lengths)[int(len(word_lengths) / 4)]}")
    print(f"Median: {sorted(word_lengths)[int(len(word_lengths) / 2)]}")
    print(f"Q3: {sorted(word_lengths)[int(len(word_lengths) * 3 / 4)]}")
    print(
        f"Max word length: {max(word_lengths_c)} (count: {word_lengths_c[max(word_lengths_c)]}) (examples: {', '.join([word for word, count in words.items() if len(word) == max(word_lengths_c)])}))")
    print(
        f"Average word length: {sum([word_length * count for word_length, count in word_lengths_c.items()]) / sum(word_lengths_c.values())}")
    print(f"Most common word lengths: {word_lengths_c.most_common(10)}")

    word_counts_per_prompt = Counter(
        [len(post['body'].split()) for post in dataset]
    )
    print(f"Most common word counts per prompt: {word_counts_per_prompt.most_common(10)}")
    print(f"Least common word counts per prompt: {word_counts_per_prompt.most_common()[:-11:-1]}")
    # for i in range(1, max(word_counts_per_prompt) + 1):
    #     print(f"{i}, {word_counts_per_prompt[i] if i in word_counts_per_prompt else 0}")

    char_counts_per_prompt = Counter(
        [len(post['body']) for post in dataset]
    )
    print(f"Most common char counts per prompt: {char_counts_per_prompt.most_common(10)}")
    print(f"Least common char counts per prompt: {char_counts_per_prompt.most_common()[:-11:-1]}")
    # for i in range(1, max(char_counts_per_prompt) + 1):
    #     print(f"{i}, {char_counts_per_prompt[i] if i in char_counts_per_prompt else 0}")