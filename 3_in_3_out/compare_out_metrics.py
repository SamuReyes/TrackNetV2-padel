import re
import matplotlib.pyplot as plt

PATH = 'D:/archivos_locales/TFG/codigo/TrackNetV2-padel/models/'

def read_data(filename):
    with open(filename, 'r') as f:
        content = f.read()

    metrics = ['recall', 'specificity', 'precision', 'accuracy', 'tpr', 'tnr', 'f1']
    data = {metric: [] for metric in metrics}

    for metric in metrics:
        data[metric] = re.findall(fr'{metric}: (\d+\.\d+)', content)

    return data

def plot_comparison(data1, data2, data3, data4, data5, title):
    epochs = range(1, len(data1['recall']) + 1)

    for metric in data1.keys():
        fig = plt.figure(figsize=(10, 5))
        plt.plot([float(x) for x in data1[metric]], label='Loss 1')
        plt.plot([float(x) for x in data2[metric]], label='Loss 2')
        plt.plot([float(x) for x in data3[metric]], label='Loss 3')
        plt.plot([float(x) for x in data4[metric]], label='Loss 4')
        plt.plot([float(x) for x in data5[metric]], label='Loss 5')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{title} - {metric} Comparison')
        plt.legend()
        #plt.show()
        fig.savefig(PATH + metric + '.jpg')

file1_data = read_data(PATH + 'modelo_val_metrics_1.txt')
file2_data = read_data(PATH + 'modelo_val_metrics_2.txt')
file3_data = read_data(PATH + 'modelo_val_metrics_3.txt')
file4_data = read_data(PATH + 'modelo_val_metrics_4.txt')
file5_data = read_data(PATH + 'modelo_val_metrics_5.txt')

plot_comparison(file1_data, file2_data, file3_data, file4_data, file5_data, 'Files')
