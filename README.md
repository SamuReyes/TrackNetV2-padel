# TrackNetV2: Efficient TrackNet

## :gear: 1. Install
### System Environment

- Windows 10
- NVIDIA RTX3060
- Python3.5.2 / git / PYQT5 / OpenCV / pandas / numpy / PyMySQL
- TensorFlow 1.13.1/keras 2.2.4/Opencv 4.1.0/CUDA 10.1

### Package
- First, you have to install cuda, cudnn and tensorflow, tutorial:
https://medium.com/@zhanwenchen/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e
        
        $ sudo apt-get install git
        $ sudo apt-get install python3-pip
        $ pip3 install pyqt5
        $ pip3 install pandas
        $ pip3 install PyMySQL
        $ pip3 install opencv-python
        $ pip3 install imutils
        $ pip3 install Pillow
        $ pip3 install piexif
        $ pip3 install -U scikit-learn
        $ pip3 install keras
        $ git clone https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2
        
## :clapper: 2. Prediction for a single video

### Generate the predicted video and the predicted labeling csv file

You can predict coordinate of shuttlecock for a single video with:

`python3 predict.py --video_name=<videoPath> --load_weights=<weightPath>`
    
Just put the video path you want to predict on option `<videoPath>` . We provide the pretrain model weights `model_33` for 3_in_1_out version, and `model906_30` for 3_in_3_out version.

### Show the predict trajectory

After predict.py, you will have the predicted labeling csv file. You can apply show_trajectory.py to generate ball's trajectory video for fancy purpose.

`python3 show_trajectory.py <input_video_path> <input_csv_path>`
#### For example:

- `<input_video_path>` = 1_01_00.mp4
- `<input_csv_path>` = 1_01_00_predict.csv

 `python3 show_trajectory.py  1_01_00.mp4 1_01_00_predict.csv` 



##  :hammer_and_wrench: 3. How to train your own data
### Step 1 : Generate the frame of video

`python3 Frame_Generator.py <videoPath> <outputFolder>`

`<videoPath>` is the video path you want to train, and `<outputFolder>` is the output directory you want to store the frames under. 

#### For example:

- `<videoPath>` = 1_01_00.mp4
- `<outputFolder>` = frame

 `python3 Frame_Generator.py 1_01_00.mp4 frame` 


### Step 2 : Preprocess the labeling csv file

We will need .csv file generated by [our labeling tool]. 


### Step 3 : Generate training data

`python3 gen_data.py --batch=<batchSize> --label=<csvFile> --frameDir=<frameDirectory> --dataDir=<npyDataDirectory>` 

`<csvFile>` is the .csv file after apply Rearrange_Label.py at second step, `<frameDirectory>` is the frame folder of the video at first step, and `<npyDataDirectory> ` is the output directory you want to store the training data under.

#### For example:

- `<batchSize>` = 1000
- `<csvFile>` = 1_01_00_ball.csv
- `<frameDirectory>` = frame
- `<npyDataDirectory>` = npy

`python3 gen_data.py --batch=1000 --label=1_01_00_ball.csv --frameDir=frame --dataDir=npy`


### Step 4 : Start training TrackNetV2

`python3 train_TrackNet.py --save_weights=<weightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>`

`<weightPath>` is TrackNetV2 weight after this training, `<npyDataDirectory>` is the directory of the .npy training data at third step, and `<toleranceValue>` means tolerance value of true positive.

#### For example:

- `<weightPath>` = mymodel
- `<npyDataDirectory>` = npy
- `<trainingEpochs>` = 30
- `<toleranceValue>` = 4

`python3 train_TrackNet.py --save_weights=mymodel --dataDir=npy --epochs=30 --tol=4`




### Step 5 : Retrain TrackNetV2 model

If you want to retrain the model, please add load_weights argument.

`python3 train_TrackNet.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>`

`<previousWeightPath>` is the model weights you had trained before.

#### For example:

- `<previousWeightPath>` = mymodel
- `<newWeightPath>` = mynewmodel
- `<npyDataDirectory>` = npy
- `<trainingEpochs>` = 10
- `<toleranceValue>` = 4

`python3 train_TrackNet.py --load_weights=mymodel --save_weights=mynewmodel --dataDir=npy --epochs=10 --tol=4`


## :notebook_with_decorative_cover: 5. Provide the performance information

`python3 accuracy.py --load_weights=<weightPath> --dataDir=<npyDataDirectory> --tol=<toleranceValue>`

`accuracy.py` provide following version:
- Number of true positive
- Number of true negative
- Number of false positive
- Number of false negative
- Accuracy
- Precision
- Recall

`<toleranceValue>` means tolerance value of true positive. `<npyDataDirectory>` is the directory of the .npy data you want to do accuracy measure.
























