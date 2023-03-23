import json
import os
from datetime import datetime
from time import sleep

import requests
from loguru import logger
from tqdm import tqdm

from stats import get_stats
from tagger import tag_recursive

FILE_PATH = 'data/reddit_data_{subredit}.json'
os.makedirs('data', exist_ok=True)

# Types of posts
# t1_	Comment
# t2_	Account
# t3_	Link
# t4_	Message
# t5_	Subreddit
# t6_	Award


def cleanup_comments(comments):
    comments_cleaned = []
    for comment in comments:
        if 'replies' in comment['data']:
            new_comment = {
                'id': comment['data'].get('id', None),
                'subreddit_id': comment['data'].get('subreddit_id', None),
                'subreddit': comment['data'].get('subreddit', None),
                'created_utc': comment['data'].get('created_utc', None),
                'author': comment['data'].get('author', None),
                'author_fullname': comment['data'].get('author_fullname', None) if comment['data'].get('author', '[deleted]') != '[deleted]' else '[deleted]',
                'parent_id': comment['data'].get('parent_id', None),
                'link_id': comment['data'].get('link_id', None),
                'permalink': comment['data'].get('permalink', None),
                'total_awards_received': comment['data'].get('total_awards_received', 0),
                'all_awardings': [
                    {
                        'coin_price': i.get('coin_price', None),
                        'coin_reward': i.get('coin_reward', None),
                        'id': i.get('id', None),
                        'icon_url': i.get('icon_url', None),
                        'description': i.get('description', None),
                        'count': i.get('count', None),
                        'name': i.get('name', None),
                        'award_type': i.get('award_type', None),
                        'award_sub_type': i.get('award_sub_type', None),
                        'subreddit_id': i.get('subreddit_id', None),
                        'subreddit_coin_reward': i.get('subreddit_coin_reward', None),
                    } for i in comment['data'].get('all_awardings', [])
                ],
                'score': comment['data'].get('score', 0),
                'body': comment['data'].get('body', None),
                'depth': comment['data'].get('depth', None),
                'controversiality': comment['data'].get('controversiality', None),
            }
            if isinstance(comment['data'].get('replies', {}), dict):
                if isinstance(comment['data'].get('replies', {}).get('data', {}), dict):
                    if isinstance(comment['data'].get('replies', {}).get('data', {}).get('children', []), list):
                        new_comment['replies'] = cleanup_comments(comment['data'].get('replies', {}).get('data', {}).get('children', []))

            if not new_comment.get('replies', []):
                new_comment['replies'] = []
            comments_cleaned.append(new_comment)
    return comments_cleaned


def process_post(post, it=0):
    if it < 3:
        try:
            if post.get('kind'):
                post = post['data']
            # Get comments
            response = requests.get(
                f'https://www.reddit.com/r/{post["subreddit"]}/comments/{post["id"]}.json',
                headers={'User-agent': 'Mozilla/5.0'}
            )
            comments = [j for i in response.json() for j in i['data'].get('children', []) if isinstance(j, dict) and j['kind'] == 't1']
            original_post = post
            post = [j for i in response.json() for j in i['data'].get('children', []) if isinstance(j, dict) and j['kind'] == 't3'][0]['data']

            post['replies'] = cleanup_comments(comments)
            new_post = {
                'id': post.get('id', None),
                'name': post.get('name', None),
                'subreddit_id': post.get('subreddit_id', None),
                'subreddit': post.get('subreddit', None),
                'created_utc': post.get('created_utc', None),
                'author': post.get('author', None),
                'author_fullname': post.get('author_fullname', None) if post.get('author', '[deleted]') != '[deleted]' else '[deleted]',
                'title': post.get('title', None),
                'selftext': post.get('selftext', None),
                'original_selftext': original_post.get('selftext', None),
                'score': post.get('score', 0),
                'over_18': post.get('over_18', None),
                'permalink': post.get('permalink', None),
                'url': post.get('url', None),
                'total_awards_received': post.get('total_awards_received', 0),
                'all_awardings': [
                    {
                        'coin_price': i.get('coin_price', None),
                        'coin_reward': i.get('coin_reward', None),
                        'id': i.get('id', None),
                        'icon_url': i.get('icon_url', None),
                        'description': i.get('description', None),
                        'count': i.get('count', None),
                        'name': i.get('name', None),
                        'award_type': i.get('award_type', None),
                        'award_sub_type': i.get('award_sub_type', None),
                        'subreddit_id': i.get('subreddit_id', None),
                        'subreddit_coin_reward': i.get('subreddit_coin_reward', None),
                    } for i in post.get('all_awardings', [])
                ],
                'is_video': post.get('is_video', None),
                'link_flair_type': post.get('link_flair_type', None),
                'spoiler': post.get('spoiler', None),
                'contest_mode': post.get('contest_mode', None),
                'link_flair_text': post.get('link_flair_text', None),
                'replies': post.get('replies', None),
                'crosspost_parent_list': [
                    process_post(i) for i in post.get('crosspost_parent_list', [])
                ],
                'poll_data': {
                    'vote_end_timestamp': post.get('poll_data', {}).get('vote_end_timestamp', None),
                    'scrape_timestamp': datetime.now().timestamp(),
                    'options': post.get('poll_data', {}).get('options', None),
                    'total_vote_count': post.get('poll_data', {}).get('total_vote_count', None),
                } if post.get('poll_data', None) else None
            }
            tag_recursive(new_post)
            return new_post
        except Exception as e:
            logger.exception(e)
            sleep(10)
            return process_post(post, it + 1)
    else:
        return None


def scrape_subreddit(subreddit, n_new_messages=1000):
    path = FILE_PATH.format(subredit=subreddit)

    new_data = []
    scraped_ids = set()

    last_save = 0
    failures = 0
    with tqdm(total=n_new_messages) as pbar:
        while len(new_data) < n_new_messages and failures < 3:
            try:
                i = 0
                while i < 3:
                    # response = requests.get(
                    #     f'https://www.reddit.com/r/{subreddit}/new.json',
                    #     headers={'User-agent': 'Mozilla/5.0'},
                    #     params={
                    #         'limit': 100,
                    #         'count': len(new_data),
                    #         'after': new_data[-1]['name'] if new_data else None,
                    #     }
                    # )
                    response = requests.get(
                        f'https://api.pushshift.io/reddit/search/submission',
                        headers={'User-agent': 'Mozilla/5.0'},
                        params={
                            'subreddit': subreddit,
                            'limit': 1000,
                            'until': round(new_data[-1]['created_utc']) if new_data else None,
                        }
                    )
                    if response.status_code == 200:
                        break
                    else:
                        i += 1
                        logger.info(f'Got status code {response.status_code} for {response.url}... Retrying after 120sec... ({i}/3)')
                        sleep(120)

                if response.status_code != 200:
                    logger.info(f'Got status code {response.status_code} for {response.url}... Breaking...')
                    break
                if len(response.json()['data']) == 0:
                    logger.info(f'No more posts to scrape...')
                    break
                for post in response.json()['data']:
                    # if post['created_utc'] < new_data[-1]['created_utc'] if new_data else datetime.now().timestamp() + 1000000:
                    if post['id'] not in scraped_ids:
                        post = process_post(post)
                        if post:
                            scraped_ids.add(post['id'])
                            new_data.append(post)
                            pbar.update(1)
                failures = 0
            except Exception as e:
                logger.exception(e)
                failures += 1

            if len(new_data) - last_save > 2000:
                last_save = len(new_data)
                with open(path, 'w') as f:
                    json.dump(new_data, f, indent=4, ensure_ascii=False)
                    logger.info(f'Saved {len(new_data)} posts to {path}.')

    with open(path, 'w') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
        logger.info(f'Saved {len(new_data)} posts to {path}.')
        os.rename(path, path.replace('.json', f'_{len(new_data)}_{datetime.now().isoformat()}.json'))
    get_stats(new_data)


scrape_subreddit('latvia', 1_000_000)
