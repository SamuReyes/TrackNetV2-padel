import subprocess
import time

path_tiempos = "D:/archivos_locales/TFG/codigo/TrackNetV2-padel/models/pruebas"

path_weights = "D:/archivos_locales/TFG/codigo/TrackNetV2-padel/models/model906_30"

path_experimento1 = "D:/archivos_locales/TFG/codigo/TrackNetV2-padel/models/pruebas/experimento_prueba_1"
path_experimento2 = "D:/archivos_locales/TFG/codigo/TrackNetV2-padel/models/pruebas/experimento_prueba_2"
path_experimento3 = "D:/archivos_locales/TFG/codigo/TrackNetV2-padel/models/pruebas/experimento_prueba_3"

path_data_train = "D:/archivos_locales/TFG/datasets/tracknetV2_padel/npy_2folds_noball/extras_test"
path_data_val = "D:/archivos_locales/TFG/datasets/tracknetV2_padel/npy_2folds_noball/extras_validacion"

comandos = [
    "python train_TrackNet3.py --load_weights=" + path_weights + " --save_weights=" + path_experimento1 + " --dataDir=" + path_data_train + " --valDir=" + path_data_val + " --epochs=2 --tol=4 --validation=1 --loss=1",
    "python train_TrackNet3.py --load_weights=" + path_weights + " --save_weights=" + path_experimento2 + " --dataDir=" + path_data_train + " --valDir=" + path_data_val + " --epochs=2 --tol=4 --validation=1 --loss=4",
    "python train_TrackNet3.py --load_weights=" + path_weights + " --save_weights=" + path_experimento3 + " --dataDir=" + path_data_train + " --valDir=" + path_data_val + " --epochs=2 --tol=4 --validation=1 --loss=5",
]

# Crear un archivo de texto para guardar los tiempos de ejecución
with open(path_tiempos + "/tiempos_ejecucion.txt", "w") as archivo_tiempos:
    for comando in comandos:
        try:
            print(f"Ejecutando: {comando}")
            start_time = time.perf_counter()
            resultado = subprocess.run(comando, shell=True, text=True, capture_output=True)
            end_time = time.perf_counter()
            tiempo_ejecucion_min = (end_time - start_time) / 60

            # Guardar el tiempo de ejecución en el archivo de texto
            archivo_tiempos.write(f"Execution time of '{comando}': {tiempo_ejecucion_min:.6f} minutes\n")

        except Exception as e:
            print(f"Error al ejecutar el comando '{comando}': {e}")
        print("\n---\n")
