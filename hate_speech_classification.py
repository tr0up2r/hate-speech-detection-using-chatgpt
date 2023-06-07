import pandas as pd
import openai
from tqdm import tqdm
import time
import backoff  # for exponential backoff

openai.api_key = "MY_API_KEY"


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def analyze_tweet(text):
    retries = 1
    sentiment = None

    while retries > 0:
        messages = [
            {"role": "system", "content": "You are an AI language model trained to analyze and detect hate speech."},
            {"role": "user", "content": f"Analyze the following text from twitter and determine if the text is: hate speech, offensive language or none of both. Consdering the context of the ENTIRE text, return only a single word, either HATE, OFFENSIVE or NEUTRAL respectively:\n{text}"}
        ]

        completion = completions_with_backoff(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=3,
            n=1,
            stop=None,
            temperature=0
        )

        response_text = completion.choices[0].message.content
        if response_text in ["HATE", "OFFENSIVE", "NEUTRAL"]:
            result = response_text
            break
        else:
            retries -= 1
            time.sleep(0.5)
    else:
        result = "neutral"

    retries = 1
    time.sleep(0.5)

    return result


if __name__ == '__main__':
    input_file = "../hate-speech-detection-using-chatgpt/csv/labeled_data_preprocessed.csv"
    df = pd.read_csv(input_file)
    df = df.sample(frac=1)

    results = []
    i = 0

    with tqdm(total = len(df)) as pbar:
        while i < len(df):
            try:
                result = analyze_tweet(df.tweet.iloc[i])
                if result == 'HATE':
                    result = 0
                elif result == 'OFFENSIVE':
                    result = 1
                else:
                    result = 2
                results.append(result)
                i += 1
                pbar.update(1)
            except:
                pass

    column = 'prediction'
    df.insert(1, column, results)

    output_file = "../hate-speech-detection-using-chatgpt/csv/labeled_data_and_prediction.csv"
    df.to_csv(output_file, index=False)
