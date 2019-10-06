import tensorflow as tf
import numpy as np
import yaml
import sys

with open('../../hyperparams.yml', 'r') as f:
    hyperparams=yaml.load(f)

sys.path.append(hyperparams['shared_scripts'])
import Transformer as model_wrapper

data_dir=hyperparams['data_dir']
models_dir=hyperparams['models_dir']

epochs=hyperparams['epochs']
batch_size=hyperparams['batch_size']

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

train_dataset=np.load(data_dir+'dataset/unsupervised/training_90.npy')
test_dataset=np.load(data_dir+'dataset/unsupervised/evaluation.npy')

model_name='Transformer_untrained'
model_utils=model_wrapper.Model(models_dir, model_name)
model=model_utils.architecture()
model_dir=model_utils.dir

train_generator=model_utils.BatchGenerator(train_dataset, batch_size)
test_generator=model_utils.BatchGenerator(test_dataset, batch_size)

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
