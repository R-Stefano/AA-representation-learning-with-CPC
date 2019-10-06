import tensorflow as tf
import numpy as np
import yaml
import sys

with open('../../hyperparams.yml', 'r') as f:
    configs=yaml.load(f)

data_dir=configs['data_dir']
model_dir=configs['models_dir']

batch_size=64
epochs=10
dataset='sst3'

sys.path.append(configs['shared_scripts'])
import Transformer_tuner as model_wrapper

base_model_name='Transformer_untrained'
model_name='tuner_secondary_'+dataset

train_dataset=np.load(data_dir+'dataset/secondary_structure/training_'+dataset+'.npy', allow_pickle=True)
test_dataset=np.load(data_dir+'dataset/secondary_structure/validating_'+dataset+'.npy', allow_pickle=True)

print('>Loading {} model'.format(base_model_name))
model_utils=model_wrapper.Model(model_dir, base_model_name, model_name)
model=model_utils.architecture()
model_dir=model_utils.dir

train_generator=model_utils.BatchGenerator(train_dataset[0], train_dataset[1], batch_size)
test_generator=model_utils.BatchGenerator(test_dataset[0], test_dataset[1], batch_size)

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

model_utils.exportModel(model)