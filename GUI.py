import tkinter as tk
from tkinter import filedialog

def browse_video():
    global video_path
    video_path = filedialog.askopenfilename(title="Select a video file", filetypes=(("Video files", "*.mp4;*.avi"), ("All files", "*.*")))

def browse_model():
    global model_path
    model_path = filedialog.askopenfilename(title="Select a TensorFlow model", filetypes=(("Model files", "*.h5"), ("All files", "*.*")))

def start_prediction():
    if video_path and model_path:
        # Aquí iría el código principal de tu script de Python que realiza la predicción
        pass

root = tk.Tk()
root.title("Video Predictor")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

video_label = tk.Label(frame, text="Video file: None")
video_label.pack()
browse_video_button = tk.Button(frame, text="Browse video", command=browse_video)
browse_video_button.pack(pady=5)

model_label = tk.Label(frame, text="Model file: None")
model_label.pack()
browse_model_button = tk.Button(frame, text="Browse model", command=browse_model)
browse_model_button.pack(pady=5)

start_button = tk.Button(frame, text="Start prediction", command=start_prediction)
start_button.pack(pady=10)

root.mainloop()