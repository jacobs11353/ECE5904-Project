# ECE5904-Project: Protein Secondary Structure Prediction

___

## Project Introduction
This project is related to the paper “DeepACLSTM: deep asymmetric convolutional long short-term memory neural models for protein secondary structure prediction” [[1]](#references). In this paper, it tries to use CNN (convolution neural network) +LSTM (long short-term neural network) for protein secondary structure. This is a classification problem based on sequence data (protein sequence).


## Dataset
The training dataset is CullPDB data set, consisting of 5534 proteins each of 39900 features.
The 5534 proteins × 39900 features can be reshaped into 5534 proteins × 700 amino acids × 57 features.

The amino acid chains are described by a 700 × 57 matrix to keep the data size consistent. The 700 denotes the peptide chain and the 57 denotes the number of features in each amino acid. When the end of a chain is reached the rest of the vector will simply be labeled as ’No Seq’ (a padding is applied).

Among the 57 features, 22 represent the primary structure (20 amino acids, 1 unknown or any amino acid, 1 'No Seq' -padding-), 22 the Protein Profiles (same as primary structure) and 9 are the secondary structure (8 possible states, 1 'No Seq' -padding-).

The proporation of each labels in the training set (cullpdb+profile_6133) is:

<img width="375" height="300" src="/images/Correct_CB6133Frequency.png">

### The format of the sequence data
<p>(N protein x 700 amino acids x 57 features)<br/>
Among all 57 features, this model only utilize the following three pars<br/>
<li>a. feature 1:[0,22): amino acid residues, with the order of 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X','NoSeq'</li>
<li>b. feature 2:[35,57): sequence profile. Note the order of amino acid residues is ACDEFGHIKLMNPQRSTVWXY and it is different from the order for amino acid residues</li>
<li>c. label: [22,31): Secondary structure labels, with the sequence of 'L', 'B', 'E', 'G', 'I', 'H', 'S', 'T','NoSeq'</li>
</p>

For a more detailed description of the dataset and for download see [[2]](#references).

## Implementation
This project was implemented using the **Keras** framework with the **Tensorflow** backend.

It contains 8 simulations:

### 1) Original data proecess method from paper

 The label of "Nonseq" part is convert to “L” and the amino acid residue is convert to A. 

 The size of the training set is (5278 sequence).

  e.g.

 LBTEH***********<br/>
 GIHLEHEL********<br/>
	    =<br/>
 LBTEH**LLLLLLLLLLL**<br/>
 GIHLEHE**LLLLLLLLL**<br/>


 After data processing, the proporation of each labels in (cullpdb+profile_6133) is:

<img width="375" height="300" src="/images/CB6133Frequency.png">

 The performance of the testing result is:

<img width="3750" height="500" src="/images/Result_cb513_original_data_0317_162epochs.png">

### 2) Nonseq elimination by substitution & unbalanced class

The "Nonseq" part is eliminated by repeating the meaningful part of the sequence to maintain its original length(700/sequence). 

The size of the training set is (5278 sequences).
 e.g.

LBTEH*************<br/>
GIHLEHEL*********<br/>
        =<br/>
LBTEH**LBTEHLBTE** <br/>
GIHLEHEL**GIHLEHEL**<br/>


 The performance of the testing result is:

<img width="3750" height="500" src="/images/Result_cb513_original_data_0702_119epochs.png">

### 3) Nonseq elimination by substitution & balanced class
 The data processing method is same as Simulation 2 except we partial increase the proporation of sequences containing minority class. The partial balance is achieved by repeating the whole 700 length sequence containing minority class.

 The size of the training set is (5918 sequences).

 
 The performance of the testing result is:

<img width="3750" height="500" src="/images/Result_cb513_original_data_0712_120epochs.png">

### 4) Break Sequence into piece to achieve fully balanced dataset

### 5) Break Sequence into piece & unbalanced dataset

### 6) Nonseq elimination by selection & unbalanced dataset

 The "Nonseq" part is eliminated by selecting the meaningful part of the sequence only. The length of each sequence is shrink from 700 to a shorter length. Then, we concatenate all vari-length sequence and cut into 700 length pieces.

 The size of the training set (1574 sequences). 
 e.g.

LBTEH************<br/>
       +  <br/>
GIHLEHEL********<br/>
       = <br/>
LBTEH**GIHLEHEL**<br/>



  The performance of the testing result is:

<img width="3750" height="500" src="/images/Result_cb513_original_data_0717_255epochs.png">

### 7) Nonseq elimination by selection & balance dataset
 The data processing method is same as Simulation 6 except we partial increase the proporation of sequences containing minority class. The partial balance is achieved by repeating the whole 700 length sequence containing minority class.

 The size of the training set is (4478 sequences).

 The performance of the testing result is:

<img width="3750" height="500" src="/images/Result_cb513_original_data_0720_169epochs.png">

### 8) Two stage classification & unbalanced dataset

 Traing: 
 In the first stage, I convert four minority classes into one "Minor" class and trained a 5-class (i.e. L,H,E,T,Minor) classification/prediction model. 
 In the second stage, the model is trained as 4-class classification/prediction model based on “B,I,G,S” class only. 

 Testing: 
 The first stage model only distinguish four major class with minor class. 
 The performance of the testing result is:

<img width="3750" height="500" src="/images/Result_cb513_original_data_0718_315epochs.png">

 The second stage model do further classification among samples treated as “minor” class after stage 1.
 The performance of the testing result is: 

 The combine result of first stage and second stage is:

<img width="3750" height="500" src="/images/Result_cb513_original_data_0719_stage_2_186_epochs.png">

## References
\[1\]: Guo, Yanbu, et al. "DeepACLSTM: deep asymmetric convolutional long short-term memory neural models for protein secondary structure prediction." BMC bioinformatics 20.1 (2019): 1-12. - https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-019-2940-0

\[2\]: http://www.princeton.edu/~jzthree/datasets/ICML2014/

\[3\]: https://github.com/GYBTA/DALSTM
