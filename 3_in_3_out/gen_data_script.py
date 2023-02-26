import os

directory_name = 'D:/archivos_locales/TFG/datasets/tracknetV2_padel'

for fold in os.listdir(directory_name):
    i = os.path.join(directory_name, fold)
    if os.path.isdir(i):
        for clip in os.listdir(i):
            j = os.path.join(i, clip)
            if os.path.isdir(j):
                print(j)
                os.system("python gen_data3.py --batch=10 --label=" + j + "/TNLabel.csv --frameDir=" + j + " --dataDir=" + j + "/data")