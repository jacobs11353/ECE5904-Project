
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
    plt.savefig("figures/"+'lstmloss06_0719(3)' +".png" , dpi=600, facecolor='w', edgecolor='w', orientation='portrait', 
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

    plt.savefig("figures/"+'lstmaccuracy06_0719(3)'+ ".png" , dpi=600, facecolor='w', edgecolor='w', orientation='portrait', 
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

    main_output = TimeDistributedDense(4, activation='softmax', name='main_output')(protein_features)    
    

    model = Model(input=[main_input, auxiliary_input], output=[main_output])
    adam = Adam(lr=0.003)
    model.compile(optimizer = adam, loss={'main_output': 'categorical_crossentropy'}, metrics=['accuracy'])#'weighted_accuracy'])
    model.summary()
    return model


def build_model_stage1():
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

    main_output = TimeDistributedDense(5, activation='softmax', name='main_output')(protein_features)    
    

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
load_file = "model/ac_LSTM_best_time_26_07_19(3).h5" # M: val_loss E: val_weighted_accuracy
checkpointer = ModelCheckpoint(filepath=load_file,verbose=1,save_best_only=True)

history=model.fit({'main_input': traindatahot, 'aux_input': trainpssm}, {'main_output': trainlabel},validation_data=({'main_input': valdatahot, 'aux_input': valpssm},{'main_output': vallabel}),
        nb_epoch=500, batch_size=44, callbacks=[checkpointer, earlyStopping], verbose=1, shuffle=True)
plotLoss(history)
model.load_weights(load_file)

# load stage 1 model
model_stage1 = build_model_stage1()
load_file_stage_1 = "model/ac_LSTM_best_time_26_07_18.h5" 
model_stage1.load_weights(load_file_stage_1)


# load stage 2 model
load_file_stage_2 = "model/ac_LSTM_best_time_26_07_19(3).h5" 
model.load_weights(load_file_stage_2)
model_stage2 = model



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
# predict stage 1
# y_pred.shape: (514, 700, 5)
y_pred = model_stage1.predict({'main_input': testdatahot, 'aux_input': testpssm})

# select the result predicted as minor label
test_stage2_hot = np.zeros(testdatahot.shape[0]*testdatahot.shape[1], dtype = int)
test_stage2_pssm = np.zeros((testpssm.shape[0]*testpssm.shape[1], testpssm.shape[2]))

index = 0
for i in range(y_pred.shape[0]):
    
    for j in range(y_pred.shape[1]):
        # select minor label part only
        if np.argmax(y_pred[i, j]) == 4:
            test_stage2_hot[index] = testdatahot[i,j]
            test_stage2_pssm[index, :] = np.copy(testpssm[i, j, :])
            index += 1

test_stage2_hot = np.array(test_stage2_hot[:index+1])
test_stage2_pssm = np.array(test_stage2_pssm[:index+1])

num_sequence = test_stage2_hot.shape[0]//700
remain = test_stage2_hot.shape[0]%700
#number of sequence is: 16 , the reamin is: 374
print("number of sequence is:", num_sequence,", the reamin is:", remain)
 

# shape (218, 700, 21)
test_pssm_sequence = np.zeros( (num_sequence+1, testpssm.shape[1], testpssm.shape[2]) )
# shape (218, 700)
test_hot_sequence = np.zeros( (num_sequence+1, testdatahot.shape[1]) )
    
cur_location = 0
    
for r in range(num_sequence):
   for c in  range(test_label.shape[1]):
       test_pssm_sequence[r, c, :] = np.copy( test_stage2_pssm[cur_location, :] )
       test_hot_sequence[r, c] =  test_stage2_hot[cur_location] 
       cur_location += 1
       
test_pssm_sequence[num_sequence,:remain,:] = np.copy( test_stage2_pssm[cur_location:, :] )
test_hot_sequence[num_sequence,:remain] = test_stage2_hot[cur_location:]     
    
# used stage 2 model to predict the new sequence
y_pred_stage2 = model_stage2.predict({'main_input': test_hot_sequence, 'aux_input': test_pssm_sequence})
y_pred_stage2_flatten = y_pred_stage2.reshape(-1,4)

target_names = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
#target_names = ['L', 'H', 'E', 'T', 'Minor']

# step one:
    # 1,3,4,6     ->  4
    # 7           ->  3
    # 2           ->  2
    # 5           ->  1
    # 0           ->  0     
# step two:        
    # 3           ->  3
    # 1           ->  1
    # 4           ->  0
    # 6           ->  2
#target_names = ['I', 'B', 'S', 'G']


###############################
# conver one-hot encoding to categorical data
y_pred_categorical_data = []
y_true_categorical_data = []

stage_2_index = 0

for index in range(y_true.shape[0]):
  for pred,label in zip(y_pred[index], y_true[index]):
    if np.sum(label) == 0:
      continue
    if np.argmax(pred) == 4:
        cur_label = np.argmax(y_pred_stage2_flatten[stage_2_index,:]) 
        if cur_label == 0: 
            y_pred_categorical_data.append( 4 )
        elif cur_label == 2:
            y_pred_categorical_data.append( 6 ) 
        else:
           y_pred_categorical_data.append( cur_label )  
        stage_2_index += 1
    else:
        cur_label = np.argmax(pred)
        if cur_label == 1:
            y_pred_categorical_data.append( 5 )
        elif cur_label == 3:
            y_pred_categorical_data.append( 7 )
        else:
            y_pred_categorical_data.append(np.argmax(pred))
    y_true_categorical_data.append(np.argmax(label))

y_pred_categorical_data = np.array(y_pred_categorical_data)
print(y_pred_categorical_data.shape)
y_true_categorical_data = np.array(y_true_categorical_data)
print(y_true_categorical_data.shape)


y_pred_categorical_data = []
for index in range(y_true.shape[0]):
  for pred in y_pred[index]:
    if np.sum(pred) == 0:
      continue
    y_pred_categorical_data.append(np.argmax(pred))
      
print( Counter(y_pred_categorical_data) )

y_pred_categorical_data = []
for pred in y_pred_stage2_flatten:
    if np.sum(pred) == 0:
      continue
    y_pred_categorical_data.append(np.argmax(pred))
      
print( Counter(y_pred_categorical_data) )

#target_names = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']

print(classification_report(y_true_categorical_data.flatten(), y_pred_categorical_data.flatten(), target_names=target_names))


# evaluate by confusion matrix
from sklearn.metrics import confusion_matrix

#print('    L    ', '  H ', '  E   ', ' T   ', '  Minor  ')
print('    L    ', '  B ', '  E   ', ' G   ', '  I  ', ' H   ', ' S   ', ' T   ')
results = confusion_matrix(y_true_categorical_data.flatten(), y_pred_categorical_data.flatten()) 
print(results)


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