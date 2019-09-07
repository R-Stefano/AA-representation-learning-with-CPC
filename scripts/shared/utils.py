import numpy as np
import yaml
import time

with open('../../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

sequence_aa=hyperparams['max_sequence_length']

batch_size=hyperparams['training']['batch_size']

stride=hyperparams['prepare_batch']['stride'] #window stride for generating sequence_length patches as input
padding=hyperparams['prepare_batch']['padding']
window_size=hyperparams['prepare_batch']['window_size'] # #window size for stride

sequence_length=hyperparams['CPC']['input_sequence_length'] #computed: ((in - w + 2p)/s)+1
num_predic_terms=hyperparams['CPC']['num_pred_terms']
num_samples=hyperparams['CPC']['num_samples'] #tot number of samples for each pred
num_samples_positive=hyperparams['CPC']['num_samples_pos']
encoding_length=hyperparams['CPC']['encoding_size'] #encoding_length=len(hyperparams['aminos'])

def prepareBatch(dataset):

    batch_s=0
    print('LENGTH DATASET:', len(dataset))
    dataset_idxs=np.arange(len(dataset))
    while (True):
        batch_e=batch_s+batch_size

        data_idxs=dataset_idxs[batch_s:batch_e]
        batch_data=dataset[data_idxs]
        
        if batch_e>=len(dataset):
            batch_s=0
            np.random.shuffle(dataset_idxs)
        else:
            batch_s=batch_e

