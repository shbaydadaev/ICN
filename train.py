import pickle
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report
import keras
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import Dense, Reshape, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dropout
LABELS= ['Normal','Abnormal','Artifact_phase_shift','Artifact_high_gain']
#LABELS= ['Normal','Abnormal','Artifacts']
file_data = "lateral_left.csv"
file_labels = "labels.csv"

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--labelbin", required=True, help="path to label binarizer")
args = vars(ap.parse_args())
BATCH_SIZE = 128
NUM_EPOCH = 100

def read_data(data_filename, label_filename):
    features = 2
    try:
        df = pd.read_csv(data_filename)
        df_labels = pd.read_csv(label_filename)
    except: FileExistsError
    impulse = []
    data = np.asarray(df)
    

    label = np.asarray(df_labels)
    mlb = MultiLabelBinarizer()
    classes = mlb.fit_transform(label)
    num_of_subclasses = len(mlb.classes_)
    for (i, label) in enumerate(mlb.classes_):
    	print("{}. {}".format(i + 1, label))

    #input size of data 100 
    input_size = data.shape[1]-125
    shape = input_size*features
    eye = data[range(0, data.shape[0], 2)]
    head = data[range(1, data.shape[0], 2)]
    
    eye50 = [item3[25:75] for item3 in eye]
    head50 = [item4[25:75] for item4 in head]
    #impulse = [eye,head] 

    for i in range(len(eye)):
        impulseEye = np.asarray(eye50[i])
        impulseHead = np.asarray(head50[i])
        impulse.append([impulseEye, impulseHead])
        
    #shape of the data 200
    # shape (11086,100,2)
    segments = np.asarray(impulse, dtype=np.float32).reshape(-1, input_size, features)
    # shape (11086,200)
    segments = segments.reshape(segments.shape[0], shape)
    return segments, classes, shape, input_size, num_of_subclasses, mlb

#read_data(file_data,file_labels)

def plot_data(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['acc'], 'r', label='Accuracy of training data')
    plt.plot(history.history['val_acc'], 'b', label='Accuracy of validation data')
    plt.plot(history.history['loss'], 'r--', label='Loss of training data')
    plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    segments_, label_, input_shape, time_period, class_num, mlb = read_data(file_data, file_labels)
    x_train, x_test, y_train, y_test = train_test_split(segments_, label_, test_size=0.15, random_state=42)
    
    #-----------------Network Architecture #1 -----------------------------
    model = Sequential()
    model.add(Reshape((time_period, 2), input_shape=(input_shape,)))
    model.add(Conv1D(50, 5, activation='relu', input_shape=(time_period, 2)))
    model.add(Conv1D(50, 5, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(100, 5, activation='relu'))
    model.add(Conv1D(100, 5, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(class_num, activation='softmax'))
    print(model.summary())
    #-----------------Network Architecture #1-----------------------------
  
    callbacks_list = [keras.callbacks.ModelCheckpoint(filepath='./data/model11086.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='acc', patience=1)]
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    History = model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, callbacks=callbacks_list, validation_split=0.15, verbose=1)
    f = open(args["labelbin"], "wb")
    f.write(pickle.dumps(mlb))


def show_confusion_matrix(validations, predictions):

    sns.set(font_scale=1.2)            
    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(12, 12))
    sns.heatmap(matrix,
                cmap='seismic',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

y_pred_test = model.predict(x_test)
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)
show_confusion_matrix(max_y_test, max_y_pred_test)
print(classification_report(max_y_test, max_y_pred_test))
plot_data(History)