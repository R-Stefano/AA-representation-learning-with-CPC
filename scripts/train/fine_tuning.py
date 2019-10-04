import tensorflow as tf
import numpy as np
import yaml
import sys

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

from tensorflow.keras.utils import Sequence


with open('../../hyperparams.yml', 'r') as f:
    configs=yaml.load(f)

data_dir=configs['data_dir']
model_dir=configs['models_dir']

batch_size=64
epochs=10

sys.path.append(configs['shared_scripts'])
import Transformer as model_wrapper
import Transformer_tuner as model_tuner_wrapper

train_dataset=np.load(data_dir+'dataset/secondary_structure/training_30_sst8.npy', allow_pickle=True)
test_dataset=np.load(data_dir+'dataset/secondary_structure/validation_sst8.npy', allow_pickle=True)

base_model_name='Transformer_untrained'
tuner_model_name='tuner_secondary'

print('>Loading {} model'.format(base_model_name))
model_utils=model_wrapper.Model(model_dir, base_model_name)
base_model=model_utils.architecture()

base_model.load_weights(model_dir+base_model_name+'/model.h5')


model_tuner_utils=model_tuner_wrapper.Model(model_utils.dir, tuner_model_name)
model_tuner=model_tuner_utils.architecture(base_model)


train_generator=model_tuner_utils.BatchGenerator(train_dataset[0], train_dataset[1], batch_size)
test_generator=model_tuner_utils.BatchGenerator(test_dataset[0], test_dataset[1], batch_size)

model_dir=model_tuner_utils.dir

callbacks=[
    tf.keras.callbacks.TensorBoard(log_dir=model_dir+'logs/', histogram_freq=1, profile_batch = 2),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=model_dir+'model_{epoch:02d}.hdf5' ,
        monitor='val_custom_accuracy', 
        load_weights_on_restart=True, 
        save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_custom_accuracy', patience=3)
]

model.fit_generator(
    generator=train_generator,
    validation_data=test_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

model.save(model_dir+'model.h5')