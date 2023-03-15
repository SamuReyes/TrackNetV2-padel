
import os

carpeta = "validation"

for n in range(1,8):
    os.system("python detect_FN.py --labelFile=D:/archivos_locales/TFG/datasets/tracknetV2_padel/images/"+ carpeta + "/" + str(n) + "/TNLabel2.csv --predictFile=D:/archivos_locales/TFG/datasets/tracknetV2_padel/images/" + carpeta + "/" + str(n) + "_predict.csv --tol=4")