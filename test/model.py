import tensorflow as tf
import numpy as np

batch_size=5
sequence_length=1104
window_size=16
num_patches=sequence_length//window_size
num_positive=num_patches//4 #half of the patches targets are true
encoding_length=1

code_size=128 #encoder output vector length
rnn_units=256
num_predic_terms=4

learning_rate=0.001

def prepareBatch_tf(batch):
    #1. Prepare input data
    sequence_patches=tf.reshape(batch, (batch_size, num_patches, window_size))

    #2. Prepare labels (are the target data real)    
    labels=tf.sequence_mask([num_positive]*batch_size, num_patches)
    labels=tf.transpose(tf.random.shuffle(tf.transpose(labels))) #shuffle which patch to fake

    #3. Prepare target data based on labels
    patches_seq=tf.range(0, num_patches) #tensor of numbers between {0, num_patches-1}
    tf.print('Patches sequence:', patches_seq)

    true_indexes=tf.random.uniform((batch_size, num_positive), maxval=num_patches,   dtype=tf.dtypes.int32)
    true_indexes_flat=tf.reshape(true_indexes, [-1])
    tf.print('Size of positive idexes', true_indexes_flat.shape)
    tf.print('Positive idexes', true_indexes_flat)
    
    t_i_f_repeated=tf.tile(true_indexes_flat, [num_predic_terms]) #repeat indexes for num_preds
    tf.print('Repeated Positive idexes', t_i_f_repeated.shape)

    t_i_f_reshaped=tf.reshape(t_i_f_repeated, [-1, num_predic_terms])
    tf.print('Reshaped Positive idexes', t_i_f_reshaped.shape)
    tf.print('Reshaped Positive idexes', t_i_f_reshaped)

    shift=tf.range(1, num_predic_terms+1)
    tf.print('Shifted values', shift)
    true_indexes_target=t_i_f_reshaped + shift
    tf.print('Positive idexes target', true_indexes_target.shape)
    tf.print('Positive idexes', true_indexes_target)





    inputData=sequence_patches
    targetData=sequence_patches


    return inputData, targetData, tf.cast(labels, tf.int64)

def prepareBatch(batch):
    #1. Prepare input data
    inputData=np.reshape(batch, (-1, num_patches, window_size, encoding_length))

    #2. Generate labels 
    labels=(np.random.normal(size=(batch_size,num_patches))>0.5)*1

    #3. Generate target data
    targetData=np.zeros((batch_size, num_patches, num_predic_terms, window_size, encoding_length), dtype=np.int8)-1

    #get index of positive examples
    positive_idxs_batch, positive_idxs_patch=np.where(labels)
    positive_idxs_patch=positive_idxs_patch+1 #get next respect to label
    
    #TODO: HANDLE END SEQUENCE
    positive_idxs_patch[positive_idxs_patch>68]=68

    for idx_batch, idx_patch in zip(positive_idxs_batch, positive_idxs_patch):
        #handle end sequence case
        target_data=np.zeros((num_predic_terms, window_size, encoding_length), dtype=np.int8)-1

        extracted_target=inputData[idx_batch, idx_patch: idx_patch+num_predic_terms]
        target_data[:extracted_target.shape[0]]=extracted_target

        targetData[idx_batch, idx_patch-1]=target_data
    
    print(targetData)
    print('Inputdata', inputData.shape)

dataset=np.random.random((batch_size*10, sequence_length, encoding_length))

prepareBatch(dataset[:10])