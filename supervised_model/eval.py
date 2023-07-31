import pandas as pd
import numpy as np
from sklearn import metrics


def performance(ground_truth_path, prediction_path, threshold):
    df_label = pd.read_csv(ground_truth_path)
    df_similarity = pd.read_csv(prediction_path)
    df = pd.merge(df_label,df_similarity, on  =['preRow','nowRow'])

    ground_truth_list = list(df.label)
    prediction_list = list(df.similarity)
    for index in range(len(prediction_list)):
        if prediction_list[index] > threshold:
            prediction_list[index] = 1
        else:
            prediction_list[index] = 0

    precision = metrics.precision_score(ground_truth_list, prediction_list)
    print(precision)


#参考
#画精确率随threshold变化的折线图， threshold从0.5开始增长，步进0.1
def draw_multiline_windows(result, x):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_ylabel('performance')
    ax.set_xlabel('window_edge')
    plt.ylim(top=1, bottom=0)
    ax.plot(x, result, linestyle='solid', marker='o', linewidth='2',)
    fig.tight_layout()
    plt.grid()
    plt.show()


performance('../data/label.csv', '../data/similarity.csv', threshold=0.81)
x = [0.5,0.6,0.7,0.8,0.9,1.0]
result = []
for i in range(len(x)):
    precision = performance('C:/Users/86137/Desktop/similarity/label.csv', 'C:/Users/86137/Desktop/similarity/similarity.csv', threshold=x[i])
    result = result + [precision]

draw_multiline_windows(result, x)
