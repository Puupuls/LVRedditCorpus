import json
import random
import re


if __name__ == '__main__':
    with open('data/subset_labeled.json', 'r', encoding='utf8') as f:
        dataset = json.load(f)

    num_positive = len([d for d in dataset if d['sentiment'] == 'positive'])
    num_negative = len([d for d in dataset if d['sentiment'] == 'negative'])
    num_neutral = len([d for d in dataset if d['sentiment'] == 'neutral'])
    print(f'Positive: {num_positive}')
    print(f'Negative: {num_negative}')
    print(f'Neutral: {num_neutral}')

    conf_mat_per_labeler_pair = {}
    labelers = list(dataset[0]['labels'].keys())
    sentiment_to_idx = {'positive': 0, 'negative': 1, 'neutral': 2}
    idx_to_sentiment = {0: 'positive', 1: 'negative', 2: 'neutral'}
    for labeler1 in labelers:
        for labeler2 in labelers:
            if labeler1 == labeler2 or (labeler2, labeler1) in conf_mat_per_labeler_pair:
                continue
            conf_mat_per_labeler_pair[(labeler1, labeler2)] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

            for d in dataset:
                label1 = d['labels'][labeler1]
                label2 = d['labels'][labeler2]
                conf_mat_per_labeler_pair[(labeler1, labeler2)][sentiment_to_idx[label1]][sentiment_to_idx[label2]] += 1

    # Save confusion matrices to csv drawing each one as matrix in a column
    with open('data/confusion_matrices.csv', 'w', encoding='utf8') as f:
        for labeler_pair in conf_mat_per_labeler_pair:
            f.write(f'{labeler_pair[0]} vs {labeler_pair[1]}\n')
            f.write(',positive,negative,neutral\n')
            for i, row in enumerate(conf_mat_per_labeler_pair[labeler_pair]):
                f.write(idx_to_sentiment[i] + ',' + ','.join([str(x) for x in row]) + '\n')
            f.write('\n')
            f.write('\n')

    conf_mat_humans_vs_auto = {}
    for labeler in labelers:
        if labeler in ['h1', 'h2']:
            continue
        conf_mat_humans_vs_auto[labeler] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for d in dataset:
            if d['labels']['h1'] == d['labels']['h2']:
                human_label = d['labels']['h1']
                auto_label = d['labels'][labeler]
                conf_mat_humans_vs_auto[labeler][sentiment_to_idx[human_label]][sentiment_to_idx[auto_label]] += 1

    # Save confusion matrices to csv drawing each one as matrix in a column
    with open('data/confusion_matrices_humans_vs_auto.csv', 'w', encoding='utf8') as f:
        for labeler in conf_mat_humans_vs_auto:
            f.write(f'humans vs {labeler}\n')
            f.write(',positive,negative,neutral\n')
            for i, row in enumerate(conf_mat_humans_vs_auto[labeler]):
                f.write(idx_to_sentiment[i] + ',' + ','.join([str(x) for x in row]) + '\n')
            f.write('\n')
            f.write('\n')