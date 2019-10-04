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

sys.path.append(configs['shared_scripts'])
import Transformer as model_wrapper

train_dataset=np.load(data_dir+'dataset/secondary_structure/training_30_sst8.npy', allow_pickle=True)
test_dataset=np.load(data_dir+'dataset/secondary_structure/validation_sst8.npy', allow_pickle=True)

base_model_name='Transformer_untrained'

model_utils=model_wrapper.Model(model_dir, base_model_name)
model=model_utils.architecture()

num_tokens=22+1
labels=8+1
batch_size=64
epochs=10

print('>Loading {} model'.format(base_model_name))
model.load_weights(model_dir+base_model_name+'/model.h5')

base_model=tf.keras.Model(
    inputs=model.get_layer('transformer_input').input,
    outputs=model.get_layer('transformer_output').output
)

base_model.trainable=False

x_input=base_model.input
x=base_model(x_input)

output=layers.Dense(labels, activation='softmax')(x)

model=tf.keras.Model(
    inputs=x_input,
    outputs=output)

def custom_loss(y_true, y_pred):
    mask=tf.cast(tf.math.greater(y_true, 0), tf.float32)

    #encode labels
    y_true_encoded=tf.one_hot(tf.cast(y_true, tf.int32), labels)

    #compute loss
    Xentropy=tf.reduce_sum(-y_true_encoded*tf.math.log(y_pred+1e-6), axis=-1)
    #mask loss for padding positions
    Xentropy_masked=tf.math.multiply(Xentropy, mask)
    Xentropy_masked_batch_loss=tf.math.reduce_mean(Xentropy_masked)

    return Xentropy_masked_batch_loss

def custom_accuracy(y_true, y_pred):
    #Generate mask
    mask=tf.math.greater(y_true, 0)

    #encode labels
    y_pred=tf.math.argmax(y_pred, axis=-1)

    #compute accuracy
    acc=tf.cast(tf.equal(tf.cast(y_true, tf.int64), y_pred), tf.float32)

    #apply mask which flats everything but it is okay
    acc=tf.boolean_mask(acc, mask)

    return tf.math.reduce_mean(acc)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.00001),
    loss=custom_loss,
    metrics=[custom_accuracy]
)

class BatchGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y= x_set, y_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        '''
        Used by fit or fit_generator to set the number of steps per epoch
        '''
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, idx):
        b_start=idx * self.batch_size
        b_end=(idx + 1) * self.batch_size
        inds = self.indices[b_start:b_end]
        batch_x = self.x[inds]
        batch_y = self.y[inds]

        return batch_x, batch_y

    def on_epoch_end(self):
        '''
        Called at the end of the epoch
        '''
        np.random.shuffle(self.indices)


train_generator=BatchGenerator(train_dataset[0], train_dataset[1], batch_size)
test_generator=BatchGenerator(test_dataset[0], test_dataset[1], batch_size)

model_dir=model_utils.dir+'fine_tune/'

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

