import numpy as np
import gzip
import h5py

path = 'C:\\Users\\Hshin-Han Hsieh\\Documents\\DeepACLSTM\\'

"""
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
    
    
    #trainhot = datahot[seq_index[:52]] #21
    #trainlabel = labels[seq_index[:52]] #8
    #trainpssm = datapssm[seq_index[:52]] #21
    
    
    #val data
    vallabel = labels[seq_index[5278:5534]] #8
    valpssm = datapssm[seq_index[5278:5534]] # 21
    valhot = datahot[seq_index[5278:5534]] #21
    
    
    train_hot = np.ones((trainhot.shape[0], trainhot.shape[1]))
    print(trainhot.shape)
    for i in range(trainhot.shape[0]):
        for j in range(trainhot.shape[1]):
            if np.sum(trainhot[i,j,:]) != 0:
                train_hot[i,j] = np.argmax(trainhot[i,j,:])
    
    
    val_hot = np.ones((valhot.shape[0], valhot.shape[1]))
    for i in range(valhot.shape[0]):
        for j in range(valhot.shape[1]):
            if np.sum(valhot[i,j,:]) != 0:
                val_hot[i,j] = np.argmax(valhot[i,j,:])
                
    return train_hot,trainpssm,trainlabel, val_hot,valpssm,vallabel

"""
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
    
    
    
    #  'L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq'
    
    ###########################
    # upsampling the train data
    ###########################
    categorical_I_data = []
    categorical_B_data = []
    categorical_G_data = []
    categorical_S_data = []
    categorical_H_data = []
    categorical_E_data = []
    categorical_L_data = []
    
    # iterate each 700*8 sequence, and upsampling by duplicate individually 
    for index in range(trainlabel.shape[0]):
        meaningful_len = 0
        for hot, label, pssm in zip(trainhot[index], trainlabel[index], trainpssm[index]):
            # if label is Noseq, start duplicte
            if np.sum(label) == 0:
                break
            # if current label is I, record
            if np.argmax(label) == 4:
                categorical_I_data.append(index)
            
            if np.argmax(label) == 1:
                categorical_B_data.append(index)
            
            if np.argmax(label) == 3:
                categorical_G_data.append(index)
                
            if np.argmax(label) == 0:
                categorical_L_data.append(index)
                
            if np.argmax(label) == 2:
                categorical_E_data.append(index)
                
            if np.argmax(label) == 5:
                categorical_H_data.append(index)
                
            if np.argmax(label) == 6:
                categorical_S_data.append(index)
    
    
    
    I = set(categorical_I_data)
    B = set(categorical_B_data)
    G = set(categorical_G_data)
    S = set(categorical_S_data)
    L = set(categorical_L_data)
    E = set(categorical_E_data)
    H = set(categorical_H_data)
    
    print(len(set(categorical_I_data)))
    print(len(set(categorical_B_data)))
    print(len(set(categorical_G_data)))
    print(len(set(categorical_S_data)))
    print(len(set(categorical_L_data)))
    print(len(set(categorical_E_data)))
    print(len(set(categorical_H_data)))
    
    AND_set = I & B & G & S# - (H&E)
    
    
    train_hot = np.ones((trainhot.shape[0], trainhot.shape[1]))
    trainlabel_up_sampling = np.copy(trainlabel)
    trainpssm_up_sampling = np.copy(trainpssm)
    print(trainhot.shape)
    
    
    # create three new list to store the duplicate sequence
    train_hot_duplicate = []
    trainlabel_duplicate = []
    trainpssm_duplicate = []
    
    
    #Q8_length = 0
    for i in range(trainhot.shape[0]):
    #    Q8_length = 0
        for j in range(trainhot.shape[1]):
            if np.sum(trainhot[i,j,:]) != 0:
                train_hot[i,j] = np.argmax(trainhot[i,j,:])
    #            Q8_length +=1
            # for 'NoSeq', upsampling with other labels
    #        else:
    #            col = j%Q8_length
    #            train_hot[i,j] = np.argmax(trainhot[i,col,:])
    #            trainlabel_up_sampling[i,j,:] = np.copy(trainlabel[i,col,:])
    #            trainpssm_up_sampling[i,j,:] = np.copy(trainpssm[i,col,:])
                
                
        if i in AND_set:
            for d in range(20):
                train_hot_duplicate.append(np.copy(train_hot[i]))
                trainlabel_duplicate.append(np.copy(trainlabel_up_sampling[i]))
                trainpssm_duplicate.append(np.copy(trainpssm_up_sampling[i]))
    
    # merge duplicate sequence with the original sequence
    trainlabel_up_sampling = np.concatenate( (trainlabel_up_sampling, trainlabel_duplicate), axis=0)
    train_hot = np.concatenate( (train_hot, train_hot_duplicate), axis=0 )
    trainpssm_up_sampling = np.concatenate( (trainpssm_up_sampling, trainpssm_duplicate), axis=0)

    
    ###########################
    # upsampling the val data
    ###########################
    #  'L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq'

    val_I_data = []
    val_B_data = []
    val_G_data = []
    val_S_data = []
    val_H_data = []
    val_E_data = []
    val_L_data = []
    
    # iterate each 700*8 sequence, and upsampling by duplicate individually 
    for index in range(vallabel.shape[0]):
        meaningful_len = 0
        for hot, label, pssm in zip(valhot[index], vallabel[index], valpssm[index]):
            # if label is Noseq, start duplicte
            if np.sum(label) == 0:
                break
            # if current label is I, record
            if np.argmax(label) == 4:
                val_I_data.append(index)
            
            if np.argmax(label) == 1:
                val_B_data.append(index)
            
            if np.argmax(label) == 3:
                val_G_data.append(index)
                
            if np.argmax(label) == 0:
                val_L_data.append(index)
                
            if np.argmax(label) == 2:
                val_E_data.append(index)
                
            if np.argmax(label) == 5:
                val_H_data.append(index)
                
            if np.argmax(label) == 6:
                val_S_data.append(index)
    
    
    
    I = set(val_I_data)
    B = set(val_B_data)
    G = set(val_G_data)
    S = set(val_S_data)
    L = set(val_L_data)
    E = set(val_E_data)
    H = set(val_H_data)
    
    print(len(set(val_I_data)))
    print(len(set(val_B_data)))
    print(len(set(val_G_data)))
    print(len(set(val_S_data)))
    print(len(set(val_L_data)))
    print(len(set(val_E_data)))
    print(len(set(val_H_data)))
    
    AND_set = I & B & G & S# - (H&E)
    
    val_hot = np.ones((valhot.shape[0], valhot.shape[1]))
    vallabel_up_sampling = np.copy(vallabel)
    valpssm_up_sampling = np.copy(valpssm)
    print(val_hot.shape)
    
    
    # create three new list to store the duplicate sequence
    val_hot_duplicate = []
    vallabel_duplicate = []
    valpssm_duplicate = []
    
    
    #Q8_length = 0
    for i in range(valhot.shape[0]):
    #    Q8_length = 0
        for j in range(valhot.shape[1]):
            if np.sum(valhot[i,j,:]) != 0:
                val_hot[i,j] = np.argmax(valhot[i,j,:])
    #            Q8_length +=1
            # for 'NoSeq', upsampling with other labels
    #        else:
    #            col = j%Q8_length
    #            train_hot[i,j] = np.argmax(trainhot[i,col,:])
    #            trainlabel_up_sampling[i,j,:] = np.copy(trainlabel[i,col,:])
    #            trainpssm_up_sampling[i,j,:] = np.copy(trainpssm[i,col,:])
                
                
        if i in AND_set:
            for d in range(20):
                val_hot_duplicate.append(np.copy(val_hot[i]))
                vallabel_duplicate.append(np.copy(vallabel_up_sampling[i]))
                valpssm_duplicate.append(np.copy(valpssm_up_sampling[i]))
    
    # merge duplicate sequence with the original sequence
    vallabel_up_sampling = np.concatenate( (vallabel_up_sampling, vallabel_duplicate), axis=0)
    val_hot = np.concatenate( (val_hot, val_hot_duplicate), axis=0 )
    valpssm_up_sampling = np.concatenate( (valpssm_up_sampling, valpssm_duplicate), axis=0)

                
    return train_hot,trainpssm_up_sampling,trainlabel_up_sampling, val_hot,valpssm_up_sampling,vallabel_up_sampling


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