import pandas as pd
from sklearn.metrics import f1_score


def f1_score_func(preds, labels):
    return f1_score(labels, preds, average='macro'), f1_score(labels, preds, average='micro')


def get_f1_score(filename):
    # read csv
    df = pd.read_csv(filename)
    df.columns = ['label', 'prediction', 'tweet']
    
    labels = list(df.label.values)
    preds = list(df.prediction.values)
    
    return f1_score_func(preds, labels)


fine_tuned_ada = 'MY_FINE_TUNED_MODEL"

print('macro, micro')
print(f'gpt-3.5-turbo: {get_f1_score("../hate-speech-detection-using-chatgpt/csv/labeled_data_and_prediction_gpt-3.5-turbo.csv")}')
print(f'fine-tuned ada: {get_f1_score(f"../hate-speech-detection-using-chatgpt/csv/labeled_data_and_prediction_{fine_tuned_ada}.csv")}')