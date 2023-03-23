import json

file = 'data/reddit_data_latvia_8169.json'


def flatten_comments_from_posts(comments):
    replies = []
    for comment in comments:
        if 'replies' in comment:
            for reply in comment['replies']:
                replies.append(reply)
                replies.extend(flatten_comments_from_posts(reply['replies']))
    return replies


def get_stats(data, print=True):
    stats = {}
    subsets = {}

    comments = flatten_comments_from_posts(data)
    stats['posts'] = len(data)
    stats['comments'] = len(comments)
    subsets['posts'] = data
    subsets['comments'] = comments

    posts_with_comments = [post for post in data if post['replies']]
    stats['posts_with_comments'] = len(posts_with_comments)
    subsets['posts_with_comments'] = posts_with_comments

    stats['average_comments_per_post'] = len(comments) / len(posts_with_comments) if posts_with_comments else 0
    stats['max_comments_per_post'] = max([len(post['replies']) for post in posts_with_comments]) if posts_with_comments else 0
    stats['median_comments_per_post'] = sorted([len(post['replies']) for post in posts_with_comments])[len(posts_with_comments) // 2] if posts_with_comments else 0

    stats['average_comment_length'] = sum([len(comment['body']) for comment in comments]) / len(comments) if comments else 0
    stats['max_comment_length'] = max([len(comment['body']) for comment in comments]) if comments else 0
    stats['median_comment_length'] = sorted([len(comment['body']) for comment in comments])[len(comments) // 2] if comments else 0

    posts_with_body = [post for post in data if post['selftext'] and 'View Poll' not in post['selftext']]
    stats['posts_with_body'] = len(posts_with_body)
    subsets['posts_with_body'] = posts_with_body

    posts_with_body_and_comments = [post for post in posts_with_comments if post['selftext']]
    stats['posts_with_body_and_comments'] = len(posts_with_body_and_comments)
    subsets['posts_with_body_and_comments'] = posts_with_body_and_comments

    question_posts = [post for post in posts_with_body_and_comments if post['link_flair_text'] and 'Jautājums' in post['link_flair_text']]
    stats['question_posts'] = len(question_posts)
    subsets['question_posts'] = question_posts

    question_posts_with_comments = [post for post in question_posts if post['replies']]
    stats['question_posts_with_comments'] = len(question_posts_with_comments)
    subsets['question_posts_with_comments'] = question_posts_with_comments

    latvian_posts = [post for post in data if post['lang'] == 'lv']
    stats['latvian_posts'] = len(latvian_posts)
    subsets['latvian_posts'] = latvian_posts
    english_posts = [post for post in data if post['lang'] == 'en']
    stats['english_posts'] = len(english_posts)
    subsets['english_posts'] = english_posts
    unknown_posts = [post for post in data if post['lang'] == 'unknown']
    stats['unknown_posts'] = len(unknown_posts)
    subsets['unknown_posts'] = unknown_posts
    other_posts = [post for post in data if post['lang'] not in ['lv', 'en', 'unknown']]
    stats['other_posts'] = len(other_posts)
    subsets['other_posts'] = other_posts

    latvian_comments = [comment for comment in comments if comment['lang'] == 'lv']
    stats['latvian_comments'] = len(latvian_comments)
    subsets['latvian_comments'] = latvian_comments
    english_comments = [comment for comment in comments if comment['lang'] == 'en']
    stats['english_comments'] = len(english_comments)
    subsets['english_comments'] = english_comments
    unknown_comments = [comment for comment in comments if comment['lang'] == 'unknown']
    stats['unknown_comments'] = len(unknown_comments)
    subsets['unknown_comments'] = unknown_comments
    other_comments = [comment for comment in comments if comment['lang'] not in ['lv', 'en', 'unknown']]
    stats['other_comments'] = len(other_comments)
    subsets['other_comments'] = other_comments

    positive_sentiment_posts = [post for post in data if post['sentiment'] == 'positive']
    stats['positive_sentiment_posts'] = len(positive_sentiment_posts)
    subsets['positive_sentiment_posts'] = positive_sentiment_posts
    neutral_sentiment_posts = [post for post in data if post['sentiment'] == 'neutral']
    stats['neutral_sentiment_posts'] = len(neutral_sentiment_posts)
    subsets['neutral_sentiment_posts'] = neutral_sentiment_posts
    negative_sentiment_posts = [post for post in data if post['sentiment'] == 'negative']
    stats['negative_sentiment_posts'] = len(negative_sentiment_posts)
    subsets['negative_sentiment_posts'] = negative_sentiment_posts
    if print:
        print(f"Posts: {stats['posts']}")
        print(f'Comments: {stats["comments"]}')
        print(f"Posts with comments: {stats['posts_with_comments']}")
        print(f"Average comments per post: {stats['average_comments_per_post']}")
        print(f"Max comments per post: {stats['max_comments_per_post']}")
        print(f"Median comments per post: {stats['median_comments_per_post']}")
        print(f"Average comment length: {stats['average_comment_length']}")
        print(f"Max comment length: {stats['max_comment_length']}")
        print(f"Median comment length: {stats['median_comment_length']}")
        print(f"Posts with body: {stats['posts_with_body']}")
        print(f"Posts with body and comments: {stats['posts_with_body_and_comments']}")
        print(f"Question posts: {stats['question_posts']}")
        print(f"Question posts with comments: {stats['question_posts_with_comments']}")

        print(f"Latvian posts: {stats['latvian_posts']}")
        print(f"English posts: {stats['english_posts']}")
        print(f"Unknown posts: {stats['unknown_posts']}")
        print(f"Other posts: {stats['other_posts']}")

        print(f"Latvian comments: {stats['latvian_comments']}")
        print(f"English comments: {stats['english_comments']}")
        print(f"Unknown comments: {stats['unknown_comments']}")
        print(f"Other comments: {stats['other_comments']}")

        print(f"Positive sentiment posts: {stats['positive_sentiment_posts']}")
        print(f"Neutral sentiment posts: {stats['neutral_sentiment_posts']}")
        print(f"Negative sentiment posts: {stats['negative_sentiment_posts']}")

    return stats, subsets


if __name__ == '__main__':
    with open(file, 'r') as f:
        data = json.load(f)
    get_stats(data)
