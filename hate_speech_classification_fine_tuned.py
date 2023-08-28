import pandas as pd
import openai
from tqdm import tqdm
import time
import backoff  # for exponential backoff

openai.api_key = "MY_API_KEY"


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


def main(model, df):
    preds = []
    i = 0
    new_df = df.copy(deep=True)
    
    batch_size = 5

    with tqdm(total = len(df)//batch_size+1) as pbar:
        while i < len(df):
            try:
                results = analyze_tweet(df.tweet.iloc[i:i+batch_size].tolist(), model)
                for rst in results:
                    print(rst.text)
                    if rst.text == 'hate':
                        preds.append(0)
                    elif rst.text == 'offensive':
                        preds.append(1)
                    elif rst.text == 'neutral':
                        preds.append(2)
                    else:  # invalid
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

    
    main("MY_FINE_TUNED_MODEL", df)
    