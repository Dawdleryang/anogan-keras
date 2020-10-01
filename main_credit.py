from __future__ import print_function

import matplotlib
#matplotlib.use('Qt5Agg')

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import argparse
import anogan_sia_encoder as anogan
from evaluations_houssam import do_roc, do_prc, do_prg, save_results

import pandas as pd


bFullSet = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--img_idx', type=int, default=10)
parser.add_argument('--label_idx', type=int, default=1)
parser.add_argument('--label_normal', type=int, default=0)
parser.add_argument('--mode', type=str, default='test', help='train, test')
parser.add_argument('--epoch', type=int, default=19)
args = parser.parse_args()

### 0. prepare data
"""
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

X_train = X_train[:,:,:,None]
X_test = X_test[:,:,:,None]

X_test_original = X_test.copy()

label_normal = args.label_normal

X_train = X_train[y_train==label_normal]
X_test = X_test[y_test==label_normal]
print ('train shape:', X_train.shape)

"""

def load_data():
    df = pd.read_csv('./credit_data/creditcard.csv')

    TEST_RATIO = 0.25
    df.sort_values('Time', inplace = True)
    TRA_INDEX = int((1-TEST_RATIO) * df.shape[0])
    train_x = df.iloc[:TRA_INDEX, 1:-2].values
    train_y = df.iloc[:TRA_INDEX, -1].values

    test_x = df.iloc[TRA_INDEX:, 1:-2].values
    test_y = df.iloc[TRA_INDEX:, -1].values


    return train_x, train_y, test_x, test_y

#import pdb; pdb.set_trace()
X_train, y_train, X_test, y_test = load_data()

X_test_original = X_test.copy()



### 1. train generator & discriminator
train_epoch = args.epoch

if args.mode == 'train':
    Model_d, Model_g, Model_e = anogan.train(256, X_train, train_epoch)




### 3. other class anomaly detection

def anomaly_detection(test_img, g=None, d=None):
    model = anogan.anomaly_detector(g=g, d=d)
    ano_score, similar_img = anogan.compute_anomaly_score(model, test_img.reshape(1, 70), iterations=500, d=d)

    # anomaly area, 255 normalization
    np_residual = test_img.reshape(70,1) - similar_img.reshape(70,1)
    np_residual = (np_residual + 2)/4

    np_residual = (255*np_residual).astype(np.uint8)
    original_x = (test_img.reshape(70,1)*127.5+127.5).astype(np.uint8)
    similar_x = (similar_img.reshape(70,1)*127.5+127.5).astype(np.uint8)

    original_x_color = cv2.cvtColor(original_x, cv2.COLOR_GRAY2BGR)
    residual_color = cv2.applyColorMap(np_residual, cv2.COLORMAP_JET)
    show = cv2.addWeighted(original_x_color, 0.3, residual_color, 0.7, 0.)

    return ano_score, original_x, similar_x, show



img_idx = args.img_idx
label_idx = args.label_idx
count = min(img_idx, len(y_test))


if bFullSet:
    test_img = X_test_original
else:
    test_img0 = X_test_original[y_test==0][:count]
    test_img1 = X_test_original[y_test==1][:count]
    test_img = np.vstack((test_img0, test_img1))
    
    
print('test_image: ', test_img.shape)


start = cv2.getTickCount()
model = anogan.anomaly_detector(g=None, d=None)
score  = anogan.compute_anomaly_score(model, test_img, iterations=100, d=None)

time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
print ('%d : done'%(img_idx), '%.2fms'%time)
print("anomaly score : ", score)


start = cv2.getTickCount()
score1 = anogan.score_from_encoder(test_img)

time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
print ('%d : done'%(img_idx), '%.2fms'%time)
print("anomaly score1 : ", score1)

if not bFullSet:
    y_test0 = y_test[y_test==0][:count]
    y_test1 = y_test[y_test==1][:count]
    y_test = np.hstack((y_test0,y_test1))

save_results(score, y_test, 'annoGan', 'crack', 'fm', '0.5', '1', 2018, 'outlier', 0.1, step=-1)
save_results(score1, y_test, 'annoGan', 'crack', 'fm', '0.5', '1', 2018, 'outlier', 0.1, step=-1)
