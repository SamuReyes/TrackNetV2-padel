import re
import matplotlib.pyplot as plt

def read_data(filename):
    with open(filename, 'r') as f:
        content = f.read()

    metrics = ['recall', 'specificity', 'precision', 'accuracy', 'tpr', 'tnr', 'f1']
    data = {metric: [] for metric in metrics}

    for metric in metrics:
        data[metric] = re.findall(fr'{metric}: (\d+\.\d+)', content)

    return data

def plot_comparison(data1, data2, title):
    epochs = range(1, len(data1['recall']) + 1)

    for metric in data1.keys():
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, [float(x) for x in data1[metric]], label='Train', marker='o')
        plt.plot(epochs, [float(x) for x in data2[metric]], label='Validation', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'{title} - {metric} Comparison')
        plt.legend()
        plt.show()

file1_data = read_data('D:/archivos_locales/TFG/codigo/TrackNetV2-padel/models/experimento5/modelo_train_metrics.txt')
file2_data = read_data('D:/archivos_locales/TFG/codigo/TrackNetV2-padel/models/experimento5/modelo_val_metrics.txt')

plot_comparison(file1_data, file2_data, 'Files')
