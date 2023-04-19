import os

directory_name = 'D:/archivos_locales/TFG/datasets/tracknetV2_padel/npy_4folds_occluded/validation'

for clip in os.listdir(directory_name):
    j = os.path.join(directory_name, clip)
    if os.path.isdir(j):
        print(j)
        os.system("python gen_data3.py --batch=4 --label=" + j + "/TNLabel.csv --frameDir=" + j + " --dataDir=" + j + "/data")