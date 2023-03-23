import json

import flask
from flask import request

from stats import get_stats

app = flask.Flask(
    __name__,
    template_folder='',
)
app.debug = True

CUR_FILE = 'data/reddit_data_latvia_8169.json'


@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/data')
def data():
    with open(CUR_FILE, 'r') as f:
        posts = json.load(f)

    stats_all, _ = get_stats(posts, print=False)

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

    stats, _ = get_stats(posts, print=False)

    return flask.jsonify({
        'posts': posts,
        'stats': stats,
        'stats_all': stats_all,
    })


if __name__ == '__main__':
    app.run()
