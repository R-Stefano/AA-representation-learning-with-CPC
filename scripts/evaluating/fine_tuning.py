import tensorflow as tf
import numpy as np
import yaml
import sys

import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models

with open('../../hyperparams.yml', 'r') as f:
    configs=yaml.load(f)

data_dir=configs['data_dir']
model_dir=configs['models_dir']

sys.path.append(configs['shared_scripts'])
import CPC_secondary_predictor as model_wrapper
import CPC as base_model

train_dataset=np.load(data_dir+'dataset/secondary_structure/training_30_sst8.npy', allow_pickle=True)
test_dataset=np.load(data_dir+'dataset/secondary_structure/validation_sst8.npy', allow_pickle=True)

base_model_name='CPC_untrained'

model_utils=base_model.Model(model_dir, base_model_name)
model=model_utils.architecture()

custom_objects={
    'custom_loss':model_utils.custom_loss,
    'custom_accuracy': model_utils.custom_accuracy
}

print('>Loading {} model'.format(base_model_name))
model=tf.keras.models.load_model(model_dir+base_model_name+'/model.h5', custom_objects=custom_objects)

rnn_output=tf.keras.Model(
    inputs=model.get_layer('encoder_input').input,
    outputs=model.get_layer('rnn').output
)

#rnn_output.trainable = False

input_test=np.random.randint(0,23, (1, 512))

x_input=layers.Input(512)
x=rnn_output(x_input)

for num_kernels in [128, 64, 32]:
    x=layers.UpSampling1D(2)(x)
    x=layers.Conv1D(num_kernels, kernel_size=9, strides=1, activation='linear', padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=layers.LeakyReLU()(x)

output=layers.Conv1D(23, kernel_size=1, strides=1, activation='softmax', padding='same')(x)

model=tf.keras.Model(
    inputs=x_input,
    outputs=output,
    name='CPC_tuner'
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy']
)


model_utils=model_wrapper.Model(model_dir, base_model_name+'_tuner')
batch_size=64
epochs=10
#model=model_utils.architecture()


train_generator=model_utils.BatchGenerator(train_dataset[0], train_dataset[1], batch_size)
test_generator=model_utils.BatchGenerator(test_dataset[0], test_dataset[1], batch_size)

model_dir=model_utils.dir

callbacks=[
    tf.keras.callbacks.TensorBoard(log_dir=model_dir+'logs/', histogram_freq=1, profile_batch = 2),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=model_dir+'model_{epoch:02d}.hdf5' ,
        monitor='val_accuracy', 
        load_weights_on_restart=True, 
        save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
]

model.fit_generator(
    generator=train_generator,
    validation_data=test_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

model.save(model_dir+'model.h5')
model.save_weights(model_dir+'model_weights.h5')