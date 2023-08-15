import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def f1_score_func(preds, labels):
    return f1_score(labels, preds, average='macro'), f1_score(labels, preds, average='micro')


def get_metric_scores(filename):
    # read csv
    df = pd.read_csv(filename)
    df.columns = ['label', 'prediction', 'tweet']

    print(f'all length: {len(df)}')
    df = df[df['prediction'] != -1]
    print(f'valid length: {len(df)}')
    print(len(df[df['label'] == 0]))
    print(len(df[(df['label'] == 0) & (df['label'] == df['prediction'])]))
    print(len(df[df['prediction'] == 0]))
    
    labels = list(df.label.values)
    preds = list(df.prediction.values)

    scores = []
    scores.append(f1_score(labels, preds, average='macro'))
    scores.append(f1_score(labels, preds, average='micro'))
    scores.append(precision_score(labels, preds, average=None))
    scores.append(recall_score(labels, preds, average=None))
    
    return scores


fine_tuned_ada = "MY_FINE_TUNED_MODEL"
filename = "../hate-speech-detection-using-chatgpt/csv/labeled_data_and_prediction_"

metrics = ['f1 macro', 'f1 micro', 'precision', 'recall']
scores = get_metric_scores(filename + fine_tuned_ada + '.csv')

for m, s in zip(metrics, scores):
    print(f'{m}: {s}')

# print(f'gpt-3.5-turbo: {get_f1_score("../hate-speech-detection-using-chatgpt/csv/labeled_data_and_prediction_gpt-3.5-turbo.csv")}')