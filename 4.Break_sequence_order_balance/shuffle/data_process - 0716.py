import numpy as np
import gzip
import h5py
from collections import Counter

path = 'C:\\Users\\Hshin-Han Hsieh\\Documents\\DeepACLSTM\\'

##############################################  
    # helper method for upsampling 
##############################################

def upsampling(vallabel, valpssm, valhot):
    # upsampling the data
    # input nparray of label, pssm and hot
    # return the upsampling version of label_upsampling, pssm_upsampling, hot_sampling

  
    val_last_non_empty_index = []
    # iterate each sequence
    for i in range(vallabel.shape[0]):
        # iterate each location for one sequence
        index = np.sum( np.count_nonzero( vallabel[i], axis=0) )
        val_last_non_empty_index.append(index)
       
    location_size = sum(val_last_non_empty_index)     
    print("total number of non_zero location is:", location_size)
    
    val_hot_temp = np.zeros((valhot.shape[0], valhot.shape[1]), dtype = int)
    val_hot = np.zeros(location_size, dtype = int)
    val_pssm = np.zeros((location_size, valpssm.shape[2]))
    val_label = np.zeros((location_size, vallabel.shape[2]))
    
    #  'L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq'
    
    #Q8_length = 0
    index = 0
    for i in range(valhot.shape[0]):
    #    Q8_length = 0
        for j in range(valhot.shape[1]):
            if np.sum(valhot[i,j,:]) != 0:
                val_hot_temp[i,j] = int(np.argmax(valhot[i,j,:]))
            else:
                break
                
        sequence_end = val_last_non_empty_index[i]
        val_hot[index: index+sequence_end] = np.copy(val_hot_temp[i, :sequence_end])
        val_pssm[index: index+sequence_end,:] = np.copy(valpssm[i, :sequence_end, :])
        val_label[index: index+sequence_end,:] = np.copy(vallabel[i, :sequence_end, :])
        index += sequence_end
    
    
    X_val = np.hstack((val_hot.reshape(-1,1), val_pssm))
    
    #Counter({5: 19152, 2: 11764, 0: 10653, 7: 6364, 6: 4547, 3: 2173, 1: 575, 4: 6})
    y_val = np.array( [np.argmax(label) for label in val_label] )
    
    print("before upsampling, the class counting result is:", Counter(y_val) )
    #from imblearn.over_sampling import SMOTE
    
    #smo = SMOTE(random_state=42)
    #X_smo_val, y_smo_val = smo.fit_sample(X_val, y_val)  
    #print( Counter(y_smo_val) )
    
    from imblearn.over_sampling import RandomOverSampler
    
    ros=RandomOverSampler(random_state=42)
    X_res,y_res=ros.fit_sample(X_val,y_val) 
    print( "after upsampling, the calss counting result is:", Counter(y_res) )
    
    
    
    # shuffle the upsampling data
    
    from sklearn.utils import shuffle
    X_res, y_res = shuffle(X_res, y_res, random_state=0)
    
    print( "after shuffle, the class counting result is:",Counter(y_res) )
    
    num_sequence = y_res.shape[0]//700
    remain = y_res.shape[0]%700
    print("number of sequence is:", num_sequence,", the reamin is:", remain)
    
    # shape (218, 700, 21)
    val_pssm_upsampling = np.zeros( (num_sequence, valpssm.shape[1], valpssm.shape[2]) )
    # shape (218, 700)
    val_hot_upsampling = np.zeros( (num_sequence, valhot.shape[1]) )
    # shape (218, 700, 8)
    val_label_upsampling = np.zeros( (num_sequence, vallabel.shape[1], vallabel.shape[2]) )
    
    
    
    def convert_to_one_hot(y, num_class):
        return np.eye(num_class)[y.reshape(-1)]
    
    y_res_one_hot = convert_to_one_hot(y_res, vallabel.shape[2])
    
    index = 0
    
    for r in range(num_sequence):
        for c in  range(vallabel.shape[1]):
            
            val_pssm_upsampling[r, c, :] = np.copy( X_res[index, 1:] )
            val_hot_upsampling[r, c] =  X_res[index, 0] 
            val_label_upsampling[r,c, :] = np.copy( y_res_one_hot[index, :] )
            index += 1
            
    print("length should be: ", y_res.shape[0] ,", the actual value is", index+remain)
    
    return val_label_upsampling, val_pssm_upsampling, val_hot_upsampling


###############################################################################
"""Data Process (Upsampling version) Function 1: Define function to read cul6133_filtered data for training"""
###############################################################################


def load_cul6133_filted():
    '''
    TRAIN data Cullpdb+profile_6133_filtered
    Test data  CB513 CASP10 CASP11
    '''   
    print("Loading train data (Cullpdb_filted)...")
    data = np.load('data/cullpdb+profile_6133.npy')#gzip.open('data/cullpdb+profile_6133_filtered.npy.gz', 'rb'))
    data = np.reshape(data, (-1, 700, 57))
     # print data.shape 
    datahot = data[:, :, 0:21]#sequence feature
    # print 'sequence feature',dataonehot[1,:3,:]
    datapssm = data[:, :, 35:56]#profile feature
    # print 'profile feature',datapssm[1,:3,:] 
    labels = data[:, :, 22:30]    # secondary struture label , 8-d
    np.random.seed(2018)
        
    # shuffle data
    num_seqs, seqlen, feature_dim = np.shape(data)
    num_classes = labels.shape[2]
    seq_index = np.arange(0, num_seqs)# 
    np.random.shuffle(seq_index)
        
    #train data
    trainhot = datahot[seq_index[:5278]] #21
    trainlabel = labels[seq_index[:5278]] #8
    trainpssm = datapssm[seq_index[:5278]] #21
         
    #val data
    vallabel = labels[seq_index[5278:5534]] #8
    valpssm = datapssm[seq_index[5278:5534]] # 21
    valhot = datahot[seq_index[5278:5534]] #21
    
    
    """
    # 1. upsampling training data
    """
    
    train_label_upsampling, train_pssm_upsampling, train_hot_upsampling = upsampling(trainlabel, trainpssm, trainhot)
    
            
    print()
    print()
    print()
    
    
    #####################################
        # train data: upsampling
        # Counter({5: 478170, 2: 278496, 0: 263421, 7: 156269, 6: 114379, 3: 54096, 1: 14769, 4: 3766})
      #####################################
      
    
    # analysis the distribution of each label of upsampling CB513 data
    # conver one-hot encoding to categorical data
    y_train_upsampling_data = []
    
    for index in range(train_label_upsampling.shape[0]):
      for label in train_label_upsampling[index]:
        if np.sum(label) == 0:
          continue
        y_train_upsampling_data.append([np.argmax(label)])
    
    y_train_upsampling_data = np.array(y_train_upsampling_data)
    print(y_train_upsampling_data.shape)
    
    print(  Counter(  y_train_upsampling_data.flatten()  )  )
    
    
    
    #####################################
        # train data: original
        # Counter({5: 378830, 2: 239476, 0: 212981, 7: 124709, 6: 91559, 3: 42816, 1: 11569, 4: 226})
    #####################################
        
    
    # analysis the distribution of each label of upsampling CB513 data
    # conver one-hot encoding to categorical data
    y_true_train_data = []
    
    for index in range(trainlabel.shape[0]):
      for label in trainlabel[index]:
        if np.sum(label) == 0:
          continue
        y_true_train_data.append([np.argmax(label)])
    
    y_true_train_data = np.array(y_true_train_data)
    print(y_true_train_data.shape)
    
    print(  Counter(  y_true_train_data.flatten()  )  )
    
 
    
    """
    # 2. upsampling val data
    """
    print()
    print()
    print()
    
    
    val_label_upsampling, val_pssm_upsampling, val_hot_upsampling = upsampling(vallabel, valpssm, valhot)
    
    print()
    print()
    print()
    
    #####################################
        # val data:  upsampling
        
    
    # analysis the distribution of each label of upsampling CB513 data
    # conver one-hot encoding to categorical data
    y_val_upsampling_data = []
    
    for index in range(val_label_upsampling.shape[0]):
      for label in val_label_upsampling[index]:
        if np.sum(label) == 0:
          continue
        y_val_upsampling_data.append([np.argmax(label)])
    
    y_val_upsampling_data = np.array(y_val_upsampling_data)
    print(y_val_upsampling_data.shape)
    
    print(  Counter(  y_val_upsampling_data.flatten()  )  )
    
    
    #####################################
        #  val data: original 
        # Counter({5: 19152, 2: 11764, 0: 10653, 7: 6364, 6: 4547, 3: 2173, 1: 575, 4: 6})
    
    #####################################
    y_true_val_data = []
    
    for index in range(vallabel.shape[0]):
        for label in vallabel[index]:
            if np.sum(label) == 0:
                continue
            y_true_val_data.append([np.argmax(label)])
    
    y_true_val_data = np.array(y_true_val_data)
    print(y_true_val_data.shape)
    
    print(  Counter(  y_true_val_data.flatten()  )  )

    return train_hot_upsampling, train_pssm_upsampling, train_label_upsampling, val_hot_upsampling, val_pssm_upsampling, val_label_upsampling

"""    
def load_cb513():
    print ("Loading Test data (CB513)...")
    CB513= np.load(path+'data/cb513+profile_split1.npy')#gzip.open('data/cb513+profile_split1.npy.gz', 'rb'))
    CB513= np.reshape(CB513,(-1,700,57))
    # print CB513.shape 
    datahot=CB513[:, :, 0:21]#sequence feature
    datapssm=CB513[:, :, 35:56]#profile feature
    
    labels = CB513[:, :, 22:30] # secondary struture label
    testhot = datahot
    testlabel = labels
    testpssm = datapssm
    
    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i,j,:]) != 0:
                test_hot[i,j] = np.argmax(testhot[i,j,:])

    return test_hot, testpssm, testlabel
    
"""
###############################################################################
"""Data Process Function (Upsampling Version) 2: Define function to read cb513+profile_split1 data for testing"""
###############################################################################
"""
def load_cb513():
    print ("Loading Test data (CB513)...")
    CB513= np.load(path+'data\\cb513+profile_split1.npy')#gzip.open('data/cb513+profile_split1.npy.gz', 'rb'))
    CB513= np.reshape(CB513,(-1,700,57))
    # print CB513.shape 
    datahot=CB513[:, :, 0:21]#sequence feature
    datapssm=CB513[:, :, 35:56]#profile feature
    
    labels = CB513[:, :, 22:30] # secondary struture label
    testhot = datahot
    testlabel = labels
    testpssm = datapssm
    
    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    testlabel_up_sampling = np.copy(testlabel)
    testpssm_up_sampling = np.copy(testpssm)
    Q8_length = 0
    for i in range(testhot.shape[0]):
        Q8_length = 0
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i,j,:]) != 0:
                test_hot[i,j] = np.argmax(testhot[i,j,:])
                Q8_length +=1
            # for 'NoSeq', upsampling with other labels
            else:
                col = j%Q8_length
                test_hot[i,j] = np.argmax(testhot[i,col,:])
                testlabel_up_sampling[i,j,:] = np.copy(testlabel[i,col,:])
                testpssm_up_sampling[i,j,:] = np.copy(testpssm[i,col,:])

    return test_hot, testpssm_up_sampling, testlabel_up_sampling
"""

def load_cb513():
    print ("Loading Test data (CB513)...")
    CB513= np.load(path+'data/cb513+profile_split1.npy')#gzip.open('data/cb513+profile_split1.npy.gz', 'rb'))
    CB513= np.reshape(CB513,(-1,700,57))
    print(CB513.shape) 
    datahot=CB513[:, :, 0:21]#sequence feature
    datapssm=CB513[:, :, 35:56]#profile feature
    
    labels = CB513[:, :, 22:30] # secondary struture label
    testhot = datahot
    testlabel = labels
    testpssm = datapssm
    
    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i,j,:]) != 0:
                test_hot[i,j] = np.argmax(testhot[i,j,:])

    return test_hot, testpssm, testlabel

def load_casp10():
    print ("Loading Test data (CASP10)...")
    casp10 = h5py.File("data/casp10.h5")
    # print casp10.shape
    datahot = casp10['features'][:, :, 0:21]#sequence feature
    datapssm =casp10['features'][:, :, 21:42]#profile feature
    labels = casp10['labels'][:, :, 0:8] # secondary struture label 
    
    testhot = datahot
    testlabel = labels
    testpssm = datapssm
    
    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i,j,:]) != 0:
                test_hot[i,j] = np.argmax(testhot[i,j,:])
    
    return test_hot, testpssm, testlabel
    
def load_casp11():
    print ("Loading Test data (CASP11)...")
    casp11 = h5py.File("data/casp11.h5")
    # print casp11.shape
    datahot=casp11['features'][:, :, 0:21]#sequence feature
    datapssm=casp11['features'][:, :, 21:42]#profile feature   
    labels = casp11['labels'][:, :, 0:8]    # secondary struture label 
    testhot = datahot
    testlabel = labels
    testpssm = datapssm    
    test_hot = np.ones((testhot.shape[0], testhot.shape[1]))
    for i in range(testhot.shape[0]):
        for j in range(testhot.shape[1]):
            if np.sum(testhot[i,j,:]) != 0:
                test_hot[i,j] = np.argmax(testhot[i,j,:])
    return test_hot, testpssm, testlabel