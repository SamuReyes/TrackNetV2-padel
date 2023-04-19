import os

directory_name = "D:/archivos_locales/TFG/datasets/tracknetV2_padel/images/test"
weights = "D:/archivos_locales/TFG/codigo/TrackNetV2-padel/models/loss3/modelo_15"

# Remove old predict files
for clip in os.listdir(directory_name):
    j = os.path.join(directory_name, clip)
    if os.path.isfile(j):
        if j[-12:-3] == "_predict.":
            os.remove(j)
        
# Create new predictions
for clip in os.listdir(directory_name):
    j = os.path.join(directory_name, clip)
    if os.path.isfile(j):
        os.system("python predict3.py --video_name=" + j + " --load_weights=" + weights)
