import tensorflow as tf
import yaml
import sys

with open('../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

sys.path.append(hyperparams['shared_scripts'])
import modelGenerator as modelGen

data_dir=hyperparams['data_dir']

learning_rate=hyperparams['training']['learning_rate']

stride=hyperparams['prepare_batch']['stride'] #window stride for generating sequence_length patches as input
padding=hyperparams['prepare_batch']['padding']
window_size=hyperparams['prepare_batch']['window_size'] # #window size for stride

sequence_length=hyperparams['CPC']['input_sequence_length'] #computed: ((in - w + 2p)/s)+1
num_predic_terms=hyperparams['CPC']['num_pred_terms']
num_samples=hyperparams['CPC']['num_samples'] #tot number of samples for each pred
num_samples_positive=hyperparams['CPC']['num_samples_pos']
encoding_length=hyperparams['CPC']['encoding_size'] #encoding_length=len(hyperparams['aminos'])
code_size=hyperparams['CPC']['code_size'] #encoder output vector length
rnn_units=hyperparams['CPC']['rnn_units']

model=modelGen.CPCModel(sequence_length, num_predic_terms, num_samples, window_size, encoding_length, code_size, rnn_units, learning_rate)
model.load_weights(data_dir+'cpc_weights.h5')
model.summary()