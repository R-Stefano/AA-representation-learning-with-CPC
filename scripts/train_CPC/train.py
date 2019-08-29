import tensorflow as tf
import numpy as np
import yaml
import utils
import sys

with open('../../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

sys.path.append(hyperparams['shared_scripts'])
import modelGenerator as modelGen

project_dir=hyperparams['project_dir']
data_dir=hyperparams['data_dir']
seed=hyperparams['seed']

np.random.seed(seed)
train_dataset=np.load(data_dir+'train_dataset.npy', allow_pickle=True)[:4]
test_dataset=np.load(data_dir+'test_dataset.npy', allow_pickle=True)[:4]

sequence_aa=hyperparams['max_sequence_length']

epochs=hyperparams['training']['epochs']
batch_size=hyperparams['training']['batch_size']
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

model.fit_generator(
    generator=utils.prepareBatch(train_dataset),
    steps_per_epoch=len(train_dataset),
    validation_data=utils.prepareBatch(test_dataset),
    validation_steps=len(test_dataset),
    epochs=epochs
)

model.save(data_dir+'cpc.h5')
model.save_weights(data_dir+'cpc_weights.h5')
#model.to_json()

