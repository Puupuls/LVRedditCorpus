import json
import random
from collections import Counter

import flask
from flask import request
try:
    from subdomains.lvreddit.stats import get_stats
except:
    from corpus_stats import get_stats

lvreddit = flask.Blueprint(
    'lvreddit',
    __name__,
    subdomain='lvreddit' if __name__ != '__main__' else None,
    template_folder='../templates/subdomains/lvreddit',
    static_folder='../../static/subdomains/lvreddit',
    static_url_path='/static',
)

CUR_FILE = 'data/reddit_data_latvia.json'
TEMPLATE_DIR='/subdomains/lvreddit/'

@lvreddit.route('/')
def index():
    return flask.render_template(TEMPLATE_DIR+'/index.html')


@lvreddit.route('/data')
def data():
    with open(CUR_FILE, 'r', encoding='utf8') as f:
        posts = json.load(f)

    stats_all, _ = get_stats(posts, print_to_console=False)

    if request.args.get('search'):
        search = request.args.get('search')
        posts = [post for post in posts if search.lower() in post['title'].lower()]

    if request.args.get('lang'):
        lang = request.args.get('lang')
        posts = [post for post in posts if post['lang'] == lang]

    if request.args.get('sentiment'):
        sentiment = request.args.get('sentiment')
        posts = [post for post in posts if post['sentiment'] == sentiment]

    if request.args.get('offset'):
        offset = int(request.args.get('offset'))
        posts = posts[offset:]

    if request.args.get('limit'):
        limit = int(request.args.get('limit'))
        posts = posts[:limit]

    stats, _ = get_stats(posts, print_to_console=False)

    return flask.jsonify({
        'posts': posts,
        'stats': stats,
        'stats_all': stats_all,
    })


@lvreddit.route('/tagger/<name>')
def tagger(name):
    if name not in ['h1', 'h2']:
        return flask.abort(404)

    with open('data/subset_labeled.json', 'r', encoding='utf8') as f:
        dataset = json.load(f)

    name_ints = [ord(a) for a in name]
    random.seed(sum(name_ints))
    random.shuffle(dataset)
    dataset.sort(key=lambda x: x.get('labels', {}).get(name, ''))

    return flask.render_template(TEMPLATE_DIR+'tagger.html', name=name, dataset=dataset)


@lvreddit.route('/tagger/<name>/save', methods=['POST'])
def tagger_save(name):
    if name not in ['h1', 'h2']:
        return flask.abort(404)

    pid = request.json.get('id')
    sentiment = request.json.get('sentiment')

    with open('data/subset_labeled.json', 'r', encoding='utf8') as f:
        dataset = json.load(f)

    for post in dataset:
        if post['id'] == pid:
            post['labels'][name] = sentiment

    with open('data/subset_labeled.json', 'w', encoding='utf8') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    return flask.jsonify({'success': True})

@lvreddit.route('/tagger/review')
def tagger_review():
    sentiment_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
    with open('data/subset_labeled.json', 'r', encoding='utf8') as f:
        dataset = json.load(f)

    labelers = []
    for post in dataset:
        for labeler in post['labels']:
            if labeler not in labelers:
                labelers.append(labeler)

    sentiment_by_labeler = {}
    for labeler in labelers:
        sentiment_by_labeler[labeler] = {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
        }
    for post in dataset:
        for labeler in post['labels']:
            sentiment_by_labeler[labeler][post['labels'][labeler]] += 1

    posts_with_largest_same_vote_count = {}
    for post in dataset:
        vote_counts = Counter(post['labels'].values())
        label, count = vote_counts.most_common(1)[0]
        if count not in posts_with_largest_same_vote_count:
            posts_with_largest_same_vote_count[count] = []
        posts_with_largest_same_vote_count[count].append(post)

    posts_with_largest_same_vote_count = sorted(
        posts_with_largest_same_vote_count.items(),
        key=lambda x: x[0],
        reverse=True,
    )
    posts_with_largest_same_vote_count = {i: j for i, j in posts_with_largest_same_vote_count}

    confusion_matrices_per_labeler = {}
    for labeler in labelers:
        confusion_matrices_per_labeler[labeler] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for post in dataset:
            if labeler not in post['labels']:
                continue
            other_labeler_sentiments = [post['labels'][l] for l in post['labels'] if l != labeler]
            if len(other_labeler_sentiments) == 0:
                continue
            other_labeler_sentiment = round(sum([sentiment_to_id[s] for s in other_labeler_sentiments]) / len(other_labeler_sentiments))
            labeler_label = sentiment_to_id[post['labels'][labeler]]
            confusion_matrices_per_labeler[labeler][other_labeler_sentiment][labeler_label] += 1

        sum_val = 0
        for i in range(3):
            for j in range(3):
                sum_val += confusion_matrices_per_labeler[labeler][i][j]
        for i in range(3):
            for j in range(3):
                confusion_matrices_per_labeler[labeler][i][j] = confusion_matrices_per_labeler[labeler][i][j] / sum_val

    return flask.render_template(
        TEMPLATE_DIR+'tagger_review.html',
        dataset=dataset,
        labelers=labelers,
        sentiment_by_labeler=sentiment_by_labeler,
        posts_with_largest_same_vote_count=posts_with_largest_same_vote_count,
        confusion_matrices_per_labeler=confusion_matrices_per_labeler,
        )


if __name__ == '__main__':
    TEMPLATE_DIR = ''
    app = flask.Flask(__name__, template_folder='')
    app.register_blueprint(lvreddit)
    app.run(debug=True, host='0.0.0.0')