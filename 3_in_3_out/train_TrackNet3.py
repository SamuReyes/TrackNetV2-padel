import numpy as np
import sys
import getopt
import os
from glob import glob
import piexif
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, array_to_img, load_img
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from TrackNet3 import TrackNet3
import keras.backend as K
from keras import optimizers
from keras.activations import *
import tensorflow as tf
import cv2
import math
import matplotlib.pyplot as plt

BATCH_SIZE = 2
HEIGHT = 288
WIDTH = 512
# HEIGHT=360
# WIDTH=640
mag = 1
sigma = 2.5

validation_metrics = 1


def outcome(y_pred, y_true, tol):
    """ Return the numbers of true positive, true negative, false positive and false negative """
    n = y_pred.shape[0]
    i = 0
    TP = TN = FP1 = FP2 = FN = 0
    while i < n:
        for j in range(3):
            if np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) == 0:
                TN += 1
            elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) == 0:
                FP2 += 1
            elif np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) > 0:
                FN += 1
            elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) > 0:
                h_pred = y_pred[i][j] * 255
                h_true = y_true[i][j] * 255
                h_pred = h_pred.astype('uint8')
                h_true = h_true.astype('uint8')
                # h_pred
                (cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for j in range(len(rects)):
                    area = rects[j][2] * rects[j][3]
                    if area > max_area:
                        max_area_idx = j
                        max_area = area
                target = rects[max_area_idx]
                (cx_pred, cy_pred) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

                # h_true
                (cnts, _) = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                max_area_idx = 0
                max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
                for j in range(len(rects)):
                    area = rects[j][2] * rects[j][3]
                    if area > max_area:
                        max_area_idx = j
                        max_area = area
                target = rects[max_area_idx]
                (cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
                dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
                if dist > tol:
                    FP1 += 1
                else:
                    TP += 1
        i += 1
    return (TP, TN, FP1, FP2, FN)


def evaluation(y_pred, y_true, tol):
    """ Return the values of accuracy, precision and recall """
    (TP, TN, FP1, FP2, FN) = outcome(y_pred, y_true, tol)
    try:
        accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
    except:
        accuracy = 0
    try:
        precision = TP / (TP + FP1 + FP2)
    except:
        precision = 0
    try:
        recall = TP / (TP + FN)
    except:
        recall = 0
    return (accuracy, precision, recall)

def compute_metrics(TP, TN, FP1, FP2, FN):
        FP = FP1 + FP2
        recall = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        accuracy = (TN + TP) / (TN + TP + FN + FP)
        tpr = TP / (TP + FN)
        tnr = FP / ( FP + TN) 
        f1 = (precision*recall) / (precision+recall)

        return [recall, specificity, precision, accuracy, tpr, tnr, f1]

try:
    (opts, args) = getopt.getopt(sys.argv[1:], '', [
        'load_weights=',
        'save_weights=',
        'dataDir=',
        'valDir=',
        'epochs=',
        'tol=',
        "validation=",
        "loss="
    ])
    if len(opts) < 6:
        raise ''
except:
    print('usage: python3 train_TrackNet3.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --valDir=<npyValidationDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue> --validation=<1 for validation 0 for no validation> --loss=<from 1 to 5>')
    print('argument --load_weights is required only if you want to retrain the model')
    exit(1)

paramCount = {
    'load_weights': 0,
    'save_weights': 0,
    'dataDir': 0,
    'valDir': 0,
    'epochs': 0,
    'tol': 0,
    'validation': 0,
    'loss': 0
}

for (opt, arg) in opts:
    if opt == '--load_weights':
        paramCount['load_weights'] += 1
        load_weights = arg
    elif opt == '--save_weights':
        paramCount['save_weights'] += 1
        save_weights = arg
    elif opt == '--dataDir':
        paramCount['dataDir'] += 1
        dataDir = arg
    elif opt == '--valDir':
        paramCount['valDir'] += 1
        valDir = arg
    elif opt == '--epochs':
        paramCount['epochs'] += 1
        epochs = int(arg)
    elif opt == '--tol':
        paramCount['tol'] += 1
        tol = int(arg)
    elif opt == '--validation':
        paramCount['validation'] += 1
        validation_metrics = int(arg)
    elif opt == '--loss':
        paramCount['loss'] += 1
        loss_function = int(arg)
    else:
        print('usage: python3 train_TrackNet3.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --valDir=<npyValidationDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue> --validation=<1 for validation 0 for no validation> --loss=<from 1 to 5>')
        print('argument --load_weights is required only if you want to retrain the model')
        exit(1)

if paramCount['save_weights'] == 0 or paramCount['dataDir'] == 0 or paramCount['valDir'] == 0 or paramCount['epochs'] == 0 or paramCount['tol'] == 0 or paramCount['validation'] == 0 or paramCount['loss'] == 0:
    print('usage: python3 train_TrackNet3.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --valDir=<npyValidationDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue> --validation=<1 for validation 0 for no validation> --loss=<from 1 to 5>')
    print('argument --load_weights is required only if you want to retrain the model')
    exit(1)

# Loss function

loss_function = 1
def custom_loss(y_true, y_pred):

    # Basada en mapas de calor
    if loss_function == 1:
        loss = 0
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        pos_mask = tf.cast(tf.equal(y_true, 1), tf.float32)
        neg_mask = tf.cast(tf.less(y_true, 1), tf.float32)
        neg_weights = tf.pow(1 - y_true, 4)

        pos_loss = -tf.math.log(tf.clip_by_value(y_pred, 1e-4, 1. - 1e-4)) * tf.pow(1 - y_pred, 2) * pos_mask
        neg_loss = -tf.math.log(tf.clip_by_value(1 - y_pred, 1e-4, 1. - 1e-4)) * tf.pow(y_pred, 2) * neg_weights * neg_mask

        num_pos = tf.reduce_sum(pos_mask)
        pos_loss = tf.reduce_sum(pos_loss)
        neg_loss = tf.reduce_sum(neg_loss)

        loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)

        return (loss)
    
    # Basada en focal loss (segunda prueba)
    elif loss_function == 2:

        gamma=2
        alpha=0.25
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)

        return K.mean(focal_loss)
    
    # Por defecto
    elif loss_function == 3:
        loss = 0
        loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))
        return (loss)
    
    # WBCE (descrita en el paper de TrackNetV2)
    elif loss_function == 4:
        w = y_pred
        pos_loss = (1 - w) ** 2 * y_true * tf.math.log(y_pred + 1e-8)
        neg_loss = w ** 2 * (1 - y_true) * tf.math.log(1 - y_pred + 1e-8)
        wbce_loss = -tf.reduce_sum(pos_loss + neg_loss)
        return (wbce_loss)
    
    # Similar a WBCE pero usada en monotrack
    elif loss_function == 5:
        w = y_pred
        loss = -K.sum((1 - w) ** 2 * y_true * K.log(K.clip(y_pred, K.epsilon(), 1 - K.epsilon())) +
                  w ** 2 * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1 - K.epsilon())))
        return loss
    
    # Usada en monotrack (dice loss y binary cross_entropy)
    elif loss_function == 6:
        from tensorflow.keras.losses import BinaryCrossentropy
        # Dice loss
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3]) + 1e-8
        denominator = tf.reduce_sum(y_true + y_pred + 0.0001, axis=[1, 2, 3]) + 1e-8
        dice_loss = 1 - tf.reduce_mean(numerator / denominator)

        # Binary cross-entropy loss
        bce_loss = tf.reduce_mean(BinaryCrossentropy(y_true, y_pred))

        # Combine the losses
        combined_loss = 0.9 * bce_loss + 0.1 * dice_loss
        return combined_loss
    
    elif loss_function == 7:
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # Asegurar que y_pred esté en el rango [epsilon, 1-epsilon]
        loss = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        return loss

# Training for the first time
if paramCount['load_weights'] == 0:
    model = TrackNet3(HEIGHT, WIDTH)
    ADADELTA = optimizers.Adadelta(learning_rate=1)
    model.compile(loss=custom_loss, optimizer=ADADELTA, metrics=['accuracy'])
# Retraining
else:
    model = load_model(load_weights, custom_objects={'custom_loss': custom_loss})

# TRAIN
r = os.path.abspath(os.path.join(dataDir))
path = glob(os.path.join(r, '*.npy'))
num = len(path) / 2
idx = np.arange(num, dtype='int') + 1
loss_list = []

# VALIDATION
val_r = os.path.abspath(os.path.join(valDir))
val_path = glob(os.path.join(val_r, '*.npy'))
val_num = len(val_path) / 2
val_idx = np.arange(val_num, dtype='int') + 1
val_loss_list = []

print('Beginning training......')
for i in range(epochs):
    print('============epoch', i+1, '================')

    """ --------- TRAIN & TRAIN LOSS --------- """
    loss = 0
    np.random.shuffle(idx)

    # Train the network
    for j in idx:
        x_train = np.load(os.path.abspath(os.path.join(dataDir, 'x_data_' + str(j) + '.npy')))
        y_train = np.load(os.path.abspath(os.path.join(dataDir, 'y_data_' + str(j) + '.npy')))
        history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)
        #############
        loss += (history.history['loss'][0])/num
        del x_train
        del y_train

    loss_list.append(loss)

    """ --------- VAL LOSS --------- """
    TP = TN = FP1 = FP2 = FN = 0
    val_loss = 0
    # Get loss for validation
    for j in val_idx:
        val_x_train = np.load(os.path.abspath(os.path.join(valDir, 'x_data_' + str(j) + '.npy')))
        val_y_train = np.load(os.path.abspath(os.path.join(valDir, 'y_data_' + str(j) + '.npy')))
        val_y_pred = model.predict(val_x_train, batch_size=BATCH_SIZE)
        #val_loss += ((tf.reduce_sum(custom_loss(val_y_train, val_y_pred), [0, 1, 2, 3])).numpy())/(BATCH_SIZE*3*HEIGHT*WIDTH)
        val_loss += custom_loss(val_y_train, val_y_pred)

        """ --------- VALIDATION METRICS (VALIDATION) --------- """
        if validation_metrics == 1:
            val_y_pred = val_y_pred > 0.5
            val_y_pred = val_y_pred.astype('float32')
            (tp, tn, fp1, fp2, fn) = outcome(val_y_pred, val_y_train, tol)
            TP += tp
            TN += tn
            FP1 += fp1
            FP2 += fp2
            FN += fn

        del val_x_train
        del val_y_train
        del val_y_pred

    val_loss_list.append(val_loss.numpy()/val_num)

    """ --------- VALIDATION METRICS (VALIDATION) --------- """
    if validation_metrics == 1:
        val_metrics = compute_metrics(max(TP,1), max(TN,1), max(FP1,1), max(FP2,1), max(FN,1))
                
        # Save metrics into a txt file
        with open(save_weights + "_val_metrics.txt", "a+") as val_metrics_out:
            val_metrics_out.write("Epoch {}:\n".format(i+1))
            val_metrics_out.write("[TP, TN, FP1, FP2, FN]: [{}, {}, {}, {}, {}]\n".format(TP, TN, FP1, FP2, FN))
            val_metrics_out.write("recall: {:.3f}\n".format(val_metrics[0]*100))
            val_metrics_out.write("specificity: {:.3f}\n".format(val_metrics[1]*100))
            val_metrics_out.write("precision: {:.3f}\n".format(val_metrics[2]*100))
            val_metrics_out.write("accuracy: {:.3f}\n".format(val_metrics[3]*100))
            val_metrics_out.write("tpr: {:.3f}\n".format(val_metrics[4]*100))
            val_metrics_out.write("tnr: {:.3f}\n".format(val_metrics[5]*100))
            val_metrics_out.write("f1: {:.3f}\n".format(val_metrics[6]*100))
            val_metrics_out.write("------------------------------------------\n")

    """ -----  VALIDATION METRICS (TRAIN) ----- """
    
    if validation_metrics == 1:
        TP = TN = FP1 = FP2 = FN = 0
        for j in idx:
            x_train = np.load(os.path.abspath(os.path.join(dataDir, 'x_data_' + str(j) + '.npy')))
            y_train = np.load(os.path.abspath(os.path.join(dataDir, 'y_data_' + str(j) + '.npy')))
            y_pred = model.predict(x_train, batch_size=BATCH_SIZE)
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype('float32')
            (tp, tn, fp1, fp2, fn) = outcome(y_pred, y_train, tol)
            TP += tp
            TN += tn
            FP1 += fp1
            FP2 += fp2
            FN += fn
            del x_train
            del y_train
            del y_pred
        
        train_metrics = compute_metrics(max(TP,1), max(TN,1), max(FP1,1), max(FP2,1), max(FN,1))

        # Save metrics into a txt file
        with open(save_weights + "_train_metrics.txt", "a+") as train_metrics_out:
            train_metrics_out.write("Epoch {}:\n".format(i+1))
            train_metrics_out.write("[TP, TN, FP1, FP2, FN]: [{}, {}, {}, {}, {}]\n".format(TP, TN, FP1, FP2, FN))
            train_metrics_out.write("recall: {:.3f}\n".format(train_metrics[0]*100))
            train_metrics_out.write("specificity: {:.3f}\n".format(train_metrics[1]*100))
            train_metrics_out.write("precision: {:.3f}\n".format(train_metrics[2]*100))
            train_metrics_out.write("accuracy: {:.3f}\n".format(train_metrics[3]*100))
            train_metrics_out.write("tpr: {:.3f}\n".format(train_metrics[4]*100))
            train_metrics_out.write("tnr: {:.3f}\n".format(train_metrics[5]*100))
            train_metrics_out.write("f1: {:.3f}\n".format(train_metrics[6]*100))
            train_metrics_out.write("------------------------------------------\n")

    # Save intermediate weights during training
    if (i + 1) % 1 == 0:
        model.save(save_weights + '_' + str(i + 1))

        # Save loss into a txt file
        with open(save_weights + "_loss.txt", "w") as loss_out:
            loss_out.write("Train loss: " + str(loss_list) + "\n")
            loss_out.write("Val loss: " + str(val_loss_list))

print('Saving weights......')
model.save(save_weights)

# Save loss into a txt file
with open(save_weights + "_loss.txt", "w") as loss_out:
   loss_out.write("Train loss: " + str(loss_list) + "\n")
   loss_out.write("Val loss: " + str(val_loss_list))

#############################
title = 'Model Loss'
plt.title(title)
plt.xlabel('epoch')
plt.ylabel('loss')
x = np.arange(1, epochs+1, 1)
plt.plot(x, loss_list)
plt.plot(x, val_loss_list)
plt.savefig(save_weights + "model_loss" + ".jpg")
#############################

print('Done......')
