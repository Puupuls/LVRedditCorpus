import json
import re
from time import sleep

from loguru import logger
import openai
from tqdm import tqdm

class Prompter:
    openai.organization = ""
    openai.api_key = ""

    prompts = [
        '"X" In the scale From Negative to neutral to positive, the sentiment of this sentence is:',
        '"X" List the most notable things that show the sentiment (no more than 2). Then state the sentiment in one word (Negative, Neutral, Positive) defaulting to Neutral',
        '"X" First List the most notable things that show the sentiment of the text (no more than 2). State what is your confidence (in percents each in new line) of this being positive, negative and neutral text.',
        '"X" Translate this text and then List the most notable things that show the sentiment that show the sentiment of the text (no more than 2). State what is your confidence (in percents each in new line) of this being positive, negative and neutral text.',
        '"X" First List the most notable things that show the sentiment of the text (no more than 2). then replace NUMBER with your sentiment prediction (between 0=negative and 10=positive) in the following sentence: "The sentiment is: NUMBER."',
        # Sentiment: 7. The sentiment is: Positive.
        # The text is not understandable as it is in an unknown language
        '"X" Translate this text and then List the most notable things that show the sentiment of the text (no more than 2). then replace NUMBER with your sentiment prediction (between 0=negative and 10=positive) in the following sentence: "The sentiment is: NUMBER."',
        # The text says "Airport." The sentiment cannot be determined as there is no indication of positive or negative feelings towards an airport. The sentiment is: N/A.
        # I'm sorry, but the text "Nenodzerties http://t.co/6ReYm5Pek2" appears to be in Latvian script and the language model used by OpenAI's GPT-3 API, which powers this AI language model, does not support translation of Latvian text.
        '"X" Finish the sentence "The sentiment is ..." with sentiment value between 0=negative and 10=positive. do not add anything else except one number.',

        '"X" On the scale from negative to neutral to positive, the sentiment of this sentence is as follows:',
        '"X" List the most notable things that show the sentiment (no more than 2). Then state the sentiment in one word (negative, neutral, positive) defaulting to neutral.',
        '"X" List the most notable things that show the sentiment of the text (no more than 2). Describe your confidence (in percent each on a new line) that this being positive, negative, and neutral text.',
        '"X" Translate this text and then list the most notable things that show the sentiment that show the sentiment of the text (no more than 2). Describe your confidence (in percent each on a new line) that this being positive, negative, and neutral text.',
        '"X" First, list the most notable things that show the sentiment of the text (no more than 2). Then replace NUMBER with your sentiment prediction (between 0=negative and 10=positive) in the following sentence: "The sentiment is: NUMBER."',
        '"X" Translate this text and then list the most notable things that show the sentiment of the text (no more than 2). Then replace NUMBER with your sentiment prediction (between 0=negative and 10=positive) in the following sentence: "The sentiment is: NUMBER."',
        '"X" Finish the sentence "The sentiment is..." with a sentiment value between 0=negative and 10=positive. Do not add anything else except one number.',

        '"X" How does this sentence make you feel? Choose one of the following: Positive, Negative, or Neutral.',
        '"X" Based on the tone of the text, what is your overall impression? Choose one of the following: Positive, Negative, or Neutral.',
        '"X" How likely is it that the author of this text has a positive or negative attitude towards the subject matter? Choose one of the following: Positive, Negative, or Neutral.',
        '"X" What is the general sentiment of this sentence? Choose one of the following: Positive, Negative, or Neutral.',
        '"X" How would you describe the author\'s overall mood or attitude in this text? Choose one of the following: Positive, Negative, or Neutral.',
        '"X" What is your overall impression of the sentiment in this sentence? Choose one of the following: Positive, Negative, or Neutral.',

        '"X" Kāda ir šī teksta emocionālā nokrāsa? Izvēlies vienu no: Pozitīva, Negatīva vai Neitrāla.',
        '"X" Pamatojoties uz teksta toni, kāds iespaids par to rodas? Izvēlies vienu no: Pozitīva, Negatīva vai Neitrāla.',
        '"X" Kādu attieksmi pauž teksta autors? Izvēlies vienu no: Pozitīva, Negatīva vai Neitrāla.',

        '"X" On a scale of 1 to 10, where 1 is highly negative and 10 is highly positive, how would you rate the sentiment displayed in this sentence?',
        '"X" Based on the tone of the text, would you categorize the overall sentiment as positive, neutral, or negative?',
        '"X" Does the author\'s language in this sentence indicate a positive, neutral, or negative sentiment?',
        '"X" What emotional tone is exhibited in this sentence? Would you categorize it as positive, neutral, or negative?',

        '"X" Skalā no 1 līdz 10, kur 1 ir ļoti negatīvs un 10 ir ļoti pozitīvs, kā vērtētu noskaņu, kas izpaužas šajā teikumā?',
        # As an AI language model, I cannot exactly determine the sentiment of the sentence as it is in Latvian language which I am not capable to understand.
        # Kā AI, man nav emociju, bet es teiktu, ka šī noskaņa varētu būt apmēram 5-6, jo tajā ir abas pozitīvas un negatīvas emocijas vienlaikus.
        # Kā AI, man nav izpratnes par emocionāliem stāvokļiem, tāpēc es nevaru novērtēt šādu noskaņu no 1 līdz 10 skalā.
        # Kā AI valodas modeļa rūpnīcas darbinieks, es nevaru sniegt personisko vērtējumu. Tā kā teikumā tiek lietots lietuviešu valodas vārds "labdien", es varu spriest, ka tas varētu būt pozitīvs sveiciens.
        # Man kā bezpersoniskam AI nav emociju, tādēļ nevaru vērtēt noskaņu.
        # 0. Kā AI, man nav emociju, bet šajā teikumā nav nekādas vērtības, jo nav sniegta nekāda precīza informācija par jautājuma detaļām un tādējādi nav iespējams novērtēt noskaņu.
        '"X" Balstoties uz teksta toni, kāda ir kopējā noskaņa - pozitīva, neitrāla vai negatīva?',
        '"X" Vai autora valoda šajā teikumā norāda uz pozitīvu, neitrālu vai negatīvu noskaņu?',
        '"X" Kāda emocionālā nokrāsa izpaužas šajā teikumā? Vai to varētu kategorizēt kā pozitīvu, neitrālu vai negatīvu?',
    ]
    num_repeats = 1

    @staticmethod
    def get_response_chat(prompt, text, temp=0.5):
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {
                    "role": "user",
                    'content': prompt.replace("X", text)
                }
            ],
            temperature=temp,
        )
        return response['choices'][0].message.content

    @staticmethod
    def process_prompt(sample, prompt_idx, key='text'):
        if 'parsable_response' not in sample:
            sample['parsable_response'] = {}
        if 'labels' not in sample:
            sample['labels'] = {}
        if 'sentiment_detailed' not in sample:
            sample['sentiment_detailed'] = {}

        prompt = Prompter.prompts[prompt_idx]

        result_labels = []
        for iter_idx in range(Prompter.num_repeats):
            if f"gpt-3.5-turbo__{prompt_idx}_{iter_idx}" in sample['sentiment_detailed']:
                resp = sample['sentiment_detailed'][f"gpt-3.5-turbo__{prompt_idx}_{iter_idx}"]
            else:
                while True:
                    try:
                        resp = Prompter.get_response_chat(prompt, sample[key])
                        break
                    except openai.error.RateLimitError:
                        logger.warning('Rate limit error, sleeping for 10 seconds')
                        sleep(10)
                        continue

            parsable_response = True
            if prompt_idx in (0, 1, 7, 8, 14, 15, 16, 17, 18, 19, 24, 25, 26):
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
                    parsable_response = False
                    resp_clean = 'neutral'
                if resp_clean not in ['negative', 'neutral', 'positive']:
                    print('Invalid response:', resp)
                    parsable_response = False
                    resp_clean = 'neutral'
            if prompt_idx in (20, 21, 22, 28, 29, 30):
                resp_clean = resp.split("\n")[-1].lower()
                is_pos = False
                is_neg = False
                is_neu = False
                if 'negatīv' in resp_clean:
                    is_neg = True
                if 'neitrāl' in resp_clean:
                    is_neu = True
                if 'pozitīv' in resp_clean:
                    is_pos = True
                if is_pos and not is_neg and not is_neu:
                    resp_clean = 'positive'
                elif is_neg and not is_pos and not is_neu:
                    resp_clean = 'negative'
                elif is_neu and not is_pos and not is_neg:
                    resp_clean = 'neutral'
                else:
                    parsable_response = False
                    resp_clean = 'neutral'
                if resp_clean not in ['negative', 'neutral', 'positive']:
                    print('Invalid response:', resp)
                    parsable_response = False
                    resp_clean = 'neutral'
            if prompt_idx in (2, 3, 9, 10):
                # Sometimes responds with only one of them
                positive = 0
                neutral = 0
                negative = 0
                try:
                    positive = int(re.search(r'positive[ sentim]*:? -? ?(\d\d?\d?)%', resp, re.IGNORECASE).group(1))
                except:
                    parsable_response = False
                    pass
                try:
                    neutral = int(re.search(r'neutral[ sentim]*:? -? ?(\d\d?\d?)%', resp, re.IGNORECASE).group(1))
                except:
                    parsable_response = False
                    pass
                try:
                    negative = int(re.search(r'negative[ sentim]*:? -? ?(\d\d?\d?)%', resp, re.IGNORECASE).group(1))
                except:
                    parsable_response = False
                    pass
                # Default to neutral
                if positive > neutral and positive > negative:
                    resp_clean = 'positive'
                elif negative > neutral and negative > positive:
                    resp_clean = 'negative'
                else:
                    resp_clean = 'neutral'
            if prompt_idx in (4, 5, 6, 11, 12, 13, 23, 27):
                try:
                    result = float(
                        re.search(r'[The ]*sentiment[ isvalueprdcton]*:? (\d+\.?\d?)\.?', resp, re.IGNORECASE).group(1))
                except:
                    try:
                        result = float(resp.strip('.'))
                    except:
                        result = 5
                        logger.warning(f'Invalid response: {resp}')
                        parsable_response = False
                if result <= 3:
                    resp_clean = 'negative'
                elif result >= 7:
                    resp_clean = 'positive'
                else:
                    resp_clean = 'neutral'

            result_labels.append(resp_clean)
            response_count = {}
            for r in result_labels:
                if r not in response_count:
                    response_count[r] = 0
                response_count[r] += 1
            max_cnt = max(response_count.values())
            resp_clean = [k for k, v in response_count.items() if v == max_cnt]
            if len(resp_clean) > 1:
                resp_clean = 'neutral'
                sample['sentiment_detailed'][f'gpt-3.5-turbo__{prompt_idx}_{iter_idx}'] = f'Indecisive: {resp_clean}'
            else:
                resp_clean = resp_clean[0]
            try:
                del sample['sentiment_detailed'][f'gpt-3.5-turbo__{prompt_idx}']
            except:
                pass
            sample['labels'][f'gpt-3.5-turbo__{prompt_idx}_{iter_idx}'] = resp_clean
            sample['sentiment_detailed'][f'gpt-3.5-turbo__{prompt_idx}_{iter_idx}'] = resp
            sample['parsable_response'][f'gpt-3.5-turbo__{prompt_idx}_{iter_idx}'] = parsable_response

    @staticmethod
    def label_sample(sample, key='text'):
        # for prompt_idx in range(len(Prompter.prompts)):
        Prompter.process_prompt(sample, 15, key)


    @staticmethod
    def get_metrics(dataset):
        prompt_metrics = {}
        for cur_prompt_idx in range(len(Prompter.prompts)):
            parsable_responses = 0
            not_parsable_responses = 0
            for i in list(range(3))+[1000]:
                cur_prompt = f'{cur_prompt_idx}_{i}'
                prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}'] = {
                    'prompt_text': Prompter.prompts[cur_prompt_idx],
                    'prompt_idx': cur_prompt_idx,
                    'prompt_repeat_idx': i,
                    'accuracy': .0,
                    'precision_avg': .0,
                    'recall_avg': .0,
                    'f1': .0,
                    'precision': {
                        'negative': .0,
                        'neutral': .0,
                        'positive': .0,
                    },
                    'recall': {
                        'negative': .0,
                        'neutral': .0,
                        'positive': .0,
                    },
                    'percent_parsable_responses': .0,
                    'count_parsable_responses': 0,
                }
                conf_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                for d in dataset:
                    if f'gpt-3.5-turbo__{cur_prompt}' not in d['labels']:
                        continue
                    sentiment_true = {'NEG': 0, 'NEU': 1, 'POZ': 2}[d['sentiment']]
                    sentiment = {'negative': 0, 'neutral': 1, 'positive': 2}[
                        d['labels'][f'gpt-3.5-turbo__{cur_prompt}']]
                    conf_matrix[sentiment_true][sentiment] += 1
                    if d['parsable_response'][f'gpt-3.5-turbo__{cur_prompt}']:
                        parsable_responses += 1
                    else:
                        not_parsable_responses += 1
                prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['percent_parsable_responses'] = parsable_responses / (
                            parsable_responses + not_parsable_responses)
                prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['count_parsable_responses'] = parsable_responses
                for i in range(3):
                    for j in range(3):
                        if i == j:
                            sent = ['negative', 'neutral', 'positive'][i]
                            if sum(conf_matrix[i]) > 0:
                                prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['precision'][sent] = conf_matrix[i][
                                                                                                        j] / sum(
                                    conf_matrix[i])
                            else:
                                prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['precision'][sent] = 0
                            if sum([conf_matrix[k][i] for k in range(3)]) > 0:
                                prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['recall'][sent] = conf_matrix[i][
                                                                                                     j] / sum(
                                    [conf_matrix[k][i] for k in range(3)])
                            else:
                                prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['recall'][sent] = 0
                if sum([sum(conf_matrix[i]) for i in range(3)]) > 0:
                    prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['accuracy'] = sum(
                        [conf_matrix[i][i] for i in range(3)]) / sum([sum(conf_matrix[i]) for i in range(3)])
                else:
                    prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['accuracy'] = 0

                prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['precision_avg'] = sum(
                    [prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['precision'][k] for k in
                     prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['precision']]) / 3
                prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['recall_avg'] = sum(
                    [prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['recall'][k] for k in
                     prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['recall']]) / 3
                if (prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['precision_avg'] +
                    prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['recall_avg']) > 0:
                    prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['f1'] = 2 * prompt_metrics[
                        f'gpt-3.5-turbo__{cur_prompt}']['precision_avg'] * prompt_metrics[
                                                                               f'gpt-3.5-turbo__{cur_prompt}'][
                                                                               'recall_avg'] / (prompt_metrics[
                                                                                                    f'gpt-3.5-turbo__{cur_prompt}'][
                                                                                                    'precision_avg'] +
                                                                                                prompt_metrics[
                                                                                                    f'gpt-3.5-turbo__{cur_prompt}'][
                                                                                                    'recall_avg'])
                else:
                    prompt_metrics[f'gpt-3.5-turbo__{cur_prompt}']['f1'] = 0
        return prompt_metrics