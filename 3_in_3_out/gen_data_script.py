import os

directory_name = 'D:/archivos_locales/TFG/datasets/tracknetV2_padel/images/test'

for clip in os.listdir(directory_name):
    j = os.path.join(directory_name, clip)
    if os.path.isdir(j):
        print(j)
        os.system("python gen_data3.py --batch=2 --label=" + j + "/TNLabel2.csv --frameDir=" + j + " --dataDir=" + j + "/data")