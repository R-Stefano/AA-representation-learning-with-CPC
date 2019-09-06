import tensorflow as tf
import numpy as np
import yaml
import sys

with open('../../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

sys.path.append(hyperparams['shared_scripts'])
import cpc as cpc_model
import utils

#Check if GPU available
print('\n-----------\n')
if hyperparams['GPU']:
    device_name = tf.test.gpu_device_name()
    if not tf.test.is_gpu_available():
        raise SystemError('GPU device not found')
    else:
        print('Found GPU at: {}'.format(device_name))
else:
    print('Using CPU')
print('\n-----------\n')
    

project_dir=hyperparams['project_dir']
data_dir=hyperparams['data_dir']
seed=hyperparams['seed']

np.random.seed(seed)
train_dataset=np.load(data_dir+'train_dataset.npy', allow_pickle=True)
test_dataset=np.load(data_dir+'test_dataset.npy', allow_pickle=True)

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

model_utils=cpc_model.Model()
model=model_utils.architecture(sequence_length, num_predic_terms, num_samples, window_size, encoding_length, code_size, rnn_units, learning_rate)

model_dir=hyperparams['models_dir']+model_utils.name
log_dir=model_dir+'logs/'

callbacks=[
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch = 3),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=model_dir+'model.{epoch:02d}-{val_loss:.2f}.hdf5' ,
        monitor='val_custom_accuracy', 
        load_weights_on_restart=True, 
        save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_custom_accuracy', patience=3)
]

model.fit_generator(
    generator=utils.prepareBatch(train_dataset),
    steps_per_epoch=len(train_dataset)//batch_size,
    validation_data=utils.prepareBatch(test_dataset),
    validation_steps=len(test_dataset)//batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

model.save(model_dir+'model.h5')
model.save_weights(model_dir+'model_weights.h5')

