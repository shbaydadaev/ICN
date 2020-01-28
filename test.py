from keras.preprocessing.image import img_to_array
from keras.models import load_model
import pandas as pd
import numpy as np
import argparse
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math 
from operator import itemgetter

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--impulse", required=True,
	help="path to input impulses")
args = vars(ap.parse_args())

model = load_model(args["model"])
mlb = pickle.loads(open(args["labelbin"], "rb").read())
print("Classifying impulse...")

try:
    impulse1 = pd.read_csv(args["impulse"])
except: FileExistsError

label = []
data = np.asarray(impulse1)
impulse = []
eye = data[range(0, data.shape[0], 2)]
head = data[range(1, data.shape[0], 2)]
print('data', data.shape)

input_size = data.shape[1] - 125
features = 2
shape = input_size*features

def detect (data, treshold =55.0):
    detected = []
    for i in range(len(data)):
        if np.abs(data[i]) > treshold:
            detected.append(i)            
    return detected

count_eye= data.shape[0]

normal=[]
abnormal=[]
art_high_gain=[]
art_phase_shift=[]
checksum=[]
eye50p = [item3[25:75] for item3 in eye]
head50p = [item4[25:75] for item4 in head]
class_predict=[]

for i in range(int(count_eye/2)):
    segments = np.asarray((eye50p[i], head50p[i]), dtype=np.float32).reshape(-1, input_size, features)
    segments = segments.reshape(segments.shape[0], shape)
    y_pred_train = model.predict(segments)[0]
    idxs = np.argsort(y_pred_train)[::-1][:-3]
    
    for (k, j) in enumerate(idxs):
        label_ = "{}: {:.2f}".format(mlb.classes_[j], y_pred_train[j]*100)
        class_predict = ([mlb.classes_[j], "{:.2f}".format(y_pred_train[j]*100)])
        label.append(label_)
        print(class_predict)
    
    if 'Normal' in class_predict[0]:
        normal.append(i)
    elif 'Abnormal' in class_predict[0]:
        abnormal.append(i)
    elif 'Artifact_phase_shift' in class_predict[0]:
        art_phase_shift.append(i)
    elif 'Artifact_high_gain' in class_predict[0]:
        art_high_gain.append(i)

print('normal: {0:2d}, abnormal: {1:2d}, art_shift: {2:2d}, art_high: {3:2d}'.format(len(normal), len(abnormal), len(art_phase_shift), len(art_high_gain)))
percentage_normal = len(normal)/int(count_eye/2)
percentage_abnormal = len(abnormal)/int(count_eye/2)
percentage_art_shift = len(art_phase_shift)/int(count_eye/2)
percentage_art_high = len(art_high_gain)/int(count_eye/2)

norm=("Normal: {:.0f} %".format(percentage_normal*100))
abnorm=('Abnormal: {:.0f} %'.format(percentage_abnormal*100))
art_shift=('Artifact_phase_shift: {:.0f} %'.format(percentage_art_shift*100))
art_high=('Artifact_high_gain: {:.0f} %'.format(percentage_art_high*100))

print ('percentage_normal', norm)
print ('percentage_abnormal', abnorm)
print ('percentage_art_shift', art_shift)
print ('percentage_art_high', art_high)

checksum.append(norm)
checksum.append(abnorm)
checksum.append(art_shift)
checksum.append(art_high)

plt.figure(figsize=(12, 6))
plt.ylim(-200, 400)
font = {'family' : 'normal','size':15}
font2 = {'family' : 'normal','weight':'bold','size':15}
plt.title('Eye impulses classification', **font2)
plt.xlabel('Time (ms)', **font)
plt.ylabel('Velocity (deg/s)', **font)
plt.grid(True)
for i in normal:
    plt.plot(eye[i], label="Eye",  color="green", linestyle='solid')
    plt.plot(head[i], label="Head", color="blue", linestyle='solid')
for i in abnormal:
    plt.plot(eye[i], label="Eye",  color="orange", linestyle='solid')
    plt.plot(head[i], label="Head", color="blue", linestyle='solid')
for i in art_high_gain:
    plt.plot(eye[i], label="Eye",  color="red", linestyle='solid')
    plt.plot(head[i], label="Head", color="blue", linestyle='solid')
for i in art_phase_shift:
    plt.plot(eye[i], label="Eye",  color="black", linestyle='solid')
    plt.plot(head[i], label="Head", color="blue", linestyle='solid')
plt.legend(checksum, prop={'size': 15})
plt.show()

