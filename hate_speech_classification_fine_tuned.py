import pandas as pd
import openai
from tqdm import tqdm
import time
import backoff  # for exponential backoff


keys_df = pd.read_csv("../hate-speech-detection-using-chatgpt/csv/keys.csv")
input_file = keys_df.input_file.values[0]
api_key = keys_df.api_key.values[0]
fine_tuned_model = keys_df.model.values[0]
openai.api_key = api_key


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def analyze_tweet(text, model):

    response = completions_with_backoff(
        model=model,
        prompt=text,
        max_tokens=1,
        temperature=0
    )
    
    result = response.choices
    
    time.sleep(3)

    return result


def main(model, labels, df):
    preds = []
    i = 0
    new_df = df.copy(deep=True)
    
    batch_size = 5

    with tqdm(total = len(df)//batch_size+1) as pbar:
        while i < len(df):
            try:
                results = analyze_tweet(df.tweet.iloc[i:i+batch_size].tolist(), model)
                for rst in results:
                    flag = True
                    for i, label in labels:
                        if rst.text == label:
                            preds.append(i)
                            flag = False
                            break
                    
                    if flag:  # invalid
                        preds.append(-1)        

                i += batch_size
                pbar.update(1)
            except Exception as e:
                print(e)
                pass
            
    column = 'prediction'
    new_df.insert(1, column, preds)

    output_file = f"../hate-speech-detection-using-chatgpt/csv/labeled_data_and_prediction_{model}.csv"
    new_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    input_file = "../hate-speech-detection-using-chatgpt/csv/labeled_data_preprocessed_without_url_test.csv"
    df = pd.read_csv(input_file)
    df = df.sample(frac=1)

    main(fine_tuned_model, df)
    