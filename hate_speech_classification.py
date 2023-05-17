import pandas as pd
import openai
from tqdm import tqdm
import time

openai.api_key = "MY_API_KEY"


def analyze_tweet(text):
    retries = 3
    sentiment = None

    while retries > 0:
        messages = [
            {"role": "system", "content": "You are an AI language model trained to analyze and detect hate speech."},
            {"role": "user", "content": f"Analyze the following text and determine if the text is: hate speech, offensive language or none of both. Return only a single word, either HATE, OFFENSIVE or NEUTRAL respectively:\n{text}"}
        ]

        completion = openai.ChatCompletion.create(
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

    retries = 3
    time.sleep(0.5)

    return result


if __name__ == '__main__':
    input_file = "../hate-speech-detection-using-chatgpt/csv/labeled_data.csv"
    df = pd.read_csv(input_file)
    df = df.sample(frac=1)
    df = df.iloc[:100]

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
            except Exception as e:
                print('API Rate Limit Error, try again')
                pass

    column = 'prediction'
    df.insert(1, column, results)

    output_file = "../hate-speech-detection-using-chatgpt/csv/labeled_data_and_prediction.csv"
    df.to_csv(output_file, index=False)
