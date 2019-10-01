import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.metrics as metrics
from tensorflow.keras.utils import Sequence 
import numpy as np


class Model():
    def __init__(self, dir_path, model_name):
        self.name=model_name
        self.dir=dir_path+self.name+'/'

    def BatchGenerator(self, x_set, y_set, batch_size):
        return BatchGenerator(x_set, y_set, batch_size)

    def architecture(self, learning_rate=0.001):
        #import CPC model and freeze

        #add training layers

        #Build model
        cpc = models.Model(inputs=x_input, outputs=out, name=self.name)

        # Compile model
        cpc.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=self.custom_loss,
            metrics=[self.custom_accuracy]
        )

        encoder_model.summary()
        #cpc_model.summary()

        return cpc

class BatchGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
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
        batch_data_x = self.x[inds]
        batch_data_y = self.y[inds]

        return batch_data_x, batch_data_y.reshape(self.batch_size, -1, 1)

    def on_epoch_end(self):
        '''
        Called at the end of the epoch
        '''
        np.random.shuffle(self.indices)