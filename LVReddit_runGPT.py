import json
import random
import re
from collections import Counter
from multiprocessing.pool import ThreadPool

from loguru import logger
from tqdm import tqdm

from GPT_prompter import Prompter
from corpus_stats import flatten_comments_from_posts, get_stats

file = 'data/LVReddit_labeled.json'

def process(d):
    while True:
        try:
            Prompter.process_prompt(d, 15, 'body')
            break
        except Exception as e:
            logger.exception(e)
            logger.error(f'Failed to get response for {d["body"]}')
    return d

if __name__ == '__main__':
    with open(file, 'r', encoding='utf8') as f:
        dataset = json.load(f)

    # Ctrl-C handler
    import signal
    running = True
    def signal_handler(sig, frame):
        global running
        running = False
        logger.info('Ctrl-C pressed')

    signal.signal(signal.SIGINT, signal_handler)

    for i in tqdm(list(range(len(dataset)//20+1))):
        samples = dataset[i*20:(i+1)*20]

        with ThreadPool(20) as pool:
            results = pool.map(process, samples)
            for j, d in enumerate(results):
                dataset[i*20+j] = d

        if i % 20 == 0 or not running:
            with open('data/LVReddit_labeled.json', 'w', encoding='utf8') as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)
            if not running:
                break

    with open('data/LVReddit_labeled.json', 'w', encoding='utf8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)



