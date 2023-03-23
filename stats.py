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


def print_stats(data):
    comments = flatten_comments_from_posts(data)
    print(f"Posts: {len(data)}")
    print(f'Comments: {len(comments)}')

    posts_with_comments = [post for post in data if post['replies']]
    print(f"Posts with comments: {len(posts_with_comments)}")

    print(f"Average comments per post: {len(comments) / len(posts_with_comments):.2f}")
    print(f"Max comments per post: {max([len(post['replies']) for post in posts_with_comments])}")
    print(f"Median comments per post: {sorted([len(post['replies']) for post in posts_with_comments])[len(posts_with_comments) // 2]}")

    print(f"Average comment length: {sum([len(comment['body']) for comment in comments]) / len(comments):.2f}")
    print(f"Max comment length: {max([len(comment['body']) for comment in comments])}")
    print(f"Median comment length: {sorted([len(comment['body']) for comment in comments])[len(comments) // 2]}")

    posts_with_body = [post for post in data if post['selftext'] and 'View Poll' not in post['selftext']]
    print(f"Posts with body: {len(posts_with_body)}")

    posts_with_body_and_comments = [post for post in posts_with_comments if post['selftext']]
    print(f"Posts with body and comments: {len(posts_with_body_and_comments)}")

    question_posts = [post for post in posts_with_body_and_comments if post['link_flair_text'] and 'Jautājums' in post['link_flair_text']]
    print(f"Question posts: {len(question_posts)}")

    question_posts_with_comments = [post for post in question_posts if post['replies']]
    print(f"Question posts with comments: {len(question_posts_with_comments)}")

    latvian_posts = [post for post in data if post['lang'] == 'lv']
    print(f"Latvian posts: {len(latvian_posts)}")
    english_posts = [post for post in data if post['lang'] == 'en']
    print(f"English posts: {len(english_posts)}")
    unknown_posts = [post for post in data if post['lang'] == 'unknown']
    print(f"Unknown posts: {len(unknown_posts)}")
    other_posts = [post for post in data if post['lang'] not in ['lv', 'en', 'unknown']]
    print(f"Other posts: {len(other_posts)}")

    latvian_comments = [comment for comment in comments if comment['lang'] == 'lv']
    print(f"Latvian comments: {len(latvian_comments)}")
    english_comments = [comment for comment in comments if comment['lang'] == 'en']
    print(f"English comments: {len(english_comments)}")
    unknown_comments = [comment for comment in comments if comment['lang'] == 'unknown']
    print(f"Unknown comments: {len(unknown_comments)}")
    other_comments = [comment for comment in comments if comment['lang'] not in ['lv', 'en', 'unknown']]
    print(f"Other comments: {len(other_comments)}")

    positive_sentiment_posts = [post for post in data if post['sentiment'] == 'positive']
    print(f"Positive sentiment posts: {len(positive_sentiment_posts)}")
    neutral_sentiment_posts = [post for post in data if post['sentiment'] == 'neutral']
    print(f"Neutral sentiment posts: {len(neutral_sentiment_posts)}")
    negative_sentiment_posts = [post for post in data if post['sentiment'] == 'negative']
    print(f"Negative sentiment posts: {len(negative_sentiment_posts)}")


if __name__ == '__main__':
    with open(file, 'r') as f:
        data = json.load(f)
    print_stats(data)
