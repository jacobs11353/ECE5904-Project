
import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Input, Embedding, LSTM, Dense, merge, Convolution2D, GRU, TimeDistributedDense, Reshape,MaxPooling2D,Convolution1D,BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping ,ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import random as rn
import pandas as pd
import tensorflow as tf
import h5py

from data_process import load_cul6133_filted, load_cb513, load_casp10, load_casp11


#from pssm import plotLoss

np.random.seed(2018)
rn.seed(2018)

def plotLoss(history):

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['trainloss', 'valloss'], loc='upper left')
    plt.savefig("figures/"+'lstmloss06_0713' +".png" , dpi=600, facecolor='w', edgecolor='w', orientation='portrait', 
                    papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)
    plt.close()

    ## PLOT CINDEX
    plt.figure()
    plt.title('model  accuracy')
    plt.ylabel('Q8 accuracy')
    plt.xlabel('epoch')
    plt.plot(history.history['acc']) #weighted_accuracy'])
    plt.plot(history.history['val_acc']) #val_weighted_accuracy'])
    plt.legend(['trainaccuracy', 'valaccuracy'], loc='upper left')

    plt.savefig("figures/"+'lstmaccuracy06_0713'+ ".png" , dpi=600, facecolor='w', edgecolor='w', orientation='portrait', 
                            papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None)

def build_model():
    # design the deepaclstm model
    main_input = Input(shape=(700,), dtype='float32', name='main_input')
    #main_input = Masking(mask_value=23)(main_input)
    x = Embedding(output_dim=21, input_dim=21, input_length=700)(main_input)
    auxiliary_input = Input(shape=(700,21), name='aux_input')  #24
    #auxiliary_input = Masking(mask_value=0)(auxiliary_input)
    print ("main_input.getshape(): ",main_input.get_shape())
    print ("auxiliary_input.get_shape(): ",auxiliary_input.get_shape())
    concat = merge([x, auxiliary_input], mode='concat', concat_axis=-1)    

    conv1_features = Convolution1D(42,1,activation='relu', border_mode='same', W_regularizer=l2(0.001))(concat)
    # print 'conv1_features shape', conv1_features.get_shape()
    conv1_features = Reshape((700, 42, 1))(conv1_features)    

    conv2_features = Convolution2D(42,3,1,activation='relu', border_mode='same', W_regularizer=l2(0.001))(conv1_features)
    # print 'conv2_features.get_shape()', conv2_features.get_shape()    

    conv2_features = Reshape((700,42*42))(conv2_features)
    conv2_features = Dropout(0.5)(conv2_features)
    conv2_features = Dense(400, activation='relu')(conv2_features)

    #, activation='tanh', inner_activation='sigmoid',dropout_W=0.5,dropout_U=0.5
    lstm_f1 = LSTM(output_dim=300,return_sequences=True,inner_activation='sigmoid',dropout_W=0.5,dropout_U=0.5)(conv2_features)
    lstm_b1 = LSTM(output_dim=300, return_sequences=True, go_backwards=True,inner_activation='sigmoid',dropout_W=0.5,dropout_U=0.5)(conv2_features)    

    lstm_f2 = LSTM(output_dim=300, return_sequences=True,inner_activation='sigmoid',dropout_W=0.5,dropout_U=0.5)(lstm_f1)
    lstm_b2 = LSTM(output_dim=300, return_sequences=True, go_backwards=True,inner_activation='sigmoid',dropout_W=0.5,dropout_U=0.5)(lstm_b1)    

    concat_features = merge([lstm_f2, lstm_b2, conv2_features], mode='concat', concat_axis=-1)    

    concat_features = Dropout(0.4)(concat_features)
    protein_features = Dense(600,activation='relu')(concat_features)
    # protein_features = TimeDistributedDense(600,activation='relu')(concat_features)
    # protein_features = TimeDistributedDense(100,activation='relu', W_regularizer=l2(0.001))(protein_features)    

    main_output = TimeDistributedDense(8, activation='softmax', name='main_output')(protein_features)    
    

    model = Model(input=[main_input, auxiliary_input], output=[main_output])
    adam = Adam(lr=0.003)
    model.compile(optimizer = adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy'])#'weighted_accuracy'])
    model.summary()
    return model


print("^^^^^^^^ start loading data ^^^^^^^^^")
# load train/validation dataset
traindatahot, trainpssm, trainlabel, valdatahot, valpssm, vallabel = load_cul6133_filted()
print()
print()
print("Train dataset:")
print("traindatahot.shape: ", traindatahot.shape)
print("trainpssm.shape: ",trainpssm.shape)
print("trainlabel.shape: ", trainlabel.shape)
print()
print("valdatahot.snhape: ", valdatahot.shape)
print("valpssm.shape: ",valpssm.shape)
print("vallabel.shape: ", vallabel.shape)
print()
print()

# load test dataset
testdatahot,testpssm,test_label = load_cb513()
#testdatahot, testpssm, test_label = load_casp10()
#testdatahot, testpssm, test_label = load_casp11()

print()
print()
print("Testing dataset:")
print("testdatahot.shape: ", testdatahot.shape)
print("testpssm.shape: ",testpssm.shape)
print("test_label.shape: ", test_label.shape)
print()
print()
model = build_model()
# print "####### look at data's shape#########"
# print traindatahot.shape, trainpssm.shape, trainlabel.shape, testdatahot.shape, testpssm.shape,testlabel.shape, valdatahot.shape,valpssm.shape,vallabel.shape
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')#val_weighted_accuracy', patience=5, verbose=1, mode='auto')
######################
# load_file = "./model/ACNN/acnn1-3-42-400-300-blstm-FC600-42-cb6133F-0.5-0.4.h5"
#################################
# load_file = "./model/ac_LSTM_best_time_17.h5" # M: weighted_accuracy E: val_weighted_accuracy
load_file = "model/ac_LSTM_best_time_26_07_13.h5" # M: val_loss E: val_weighted_accuracy
checkpointer = ModelCheckpoint(filepath=load_file,verbose=1,save_best_only=True)

history=model.fit({'main_input': traindatahot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': valdatahot, 'aux_input': valpssm},{'main_output': vallabel}),
        nb_epoch=300, batch_size=44, callbacks=[checkpointer, earlyStopping], verbose=1, shuffle=True)
plotLoss(history)
model.load_weights(load_file)
print ("#########evaluate:##############")
#score = model.evaluate({'main_input': testdatahot, 'aux_input': testpssm},{'main_output': test_label}, verbose=2, batch_size=2)
#print (score) 
#print ('test loss:', score[0])
#print ('test accuracy:', score[1])


##########################################
  # evaluate without Noseq part
##########################################

from sklearn.metrics import classification_report
# generate classification result report
y_true = test_label

print("======= calculate the precision and recall rate ======")
y_pred = model.predict({'main_input': testdatahot, 'aux_input': testpssm})
target_names = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']



###############################
# conver one-hot encoding to categorical data
y_pred_categorical_data = []
y_true_categorical_data = []

for index in range(y_true.shape[0]):
  for pred,label in zip(y_pred[index], y_true[index]):
    if np.sum(label) == 0:
      continue
    y_pred_categorical_data.append([np.argmax(pred)])
    y_true_categorical_data.append([np.argmax(label)])

y_pred_categorical_data = np.array(y_pred_categorical_data)
print(y_pred_categorical_data.shape)
y_true_categorical_data = np.array(y_true_categorical_data)
print(y_true_categorical_data.shape)


#target_names = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']

print(classification_report(y_true_categorical_data.flatten(), y_pred_categorical_data.flatten(), target_names=target_names))




# evaluate by confusion matrix
from sklearn.metrics import confusion_matrix

 
#results = confusion_matrix(y_true_categorical_data.flatten(), y_pred_categorical_data.flatten()) 
#print(results)

y_true = y_true_categorical_data.flatten()
y_pred = y_pred_categorical_data.flatten()

import pandas as pd

unique_label = np.unique([y_true, y_pred])

cmtx = pd.DataFrame(
    confusion_matrix(y_true, y_pred, labels=unique_label), 
    index=['true:{:}'.format(target_names[x]) for x in unique_label], 
    columns=['pred:{:}'.format(target_names[x]) for x in unique_label]
)
print(cmtx)