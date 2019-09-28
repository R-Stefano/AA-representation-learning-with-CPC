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

        self.embedding_size=256
        self.sequence_length=512

    def architecture(self, learning_rate=0.001):
        x_input=layers.Input((self.sequence_length), name='input_layer')

        x=layers.Embedding(input_dim=20, output_dim=5, mask_zero=True)(x_input)

        for num_k in [32, 32, 64, 64, 128, 128]:
            x=layers.Conv1D(filters=num_k, kernel_size=9, strides=2, activation='relu')(x)

        x=layers.Flatten()(x)
        mean=layers.Dense(units=self.embedding_size, activation='linear', name='embed_mean')(x)
        std=tf.math.exp(layers.Dense(units=self.embedding_size, activation='linear', name='embed_std')(x))

        z=tf.math.add(mean, std*tf.random.normal(shape=[self.embedding_size]), name='embed_code')

        x=tf.expand_dims(z, axis=1)

        for num_k, upsampling in zip([256, 256, 128, 128, 64, 32], [2, 4, 2, 4, 2, 2]):
            x=layers.UpSampling1D(upsampling)(x)
            x=layers.Conv1D(filters=num_k, kernel_size=9, activation='relu', padding='same')(x)
        
        x=layers.UpSampling1D()(x)
        output=layers.Conv1D(filters=21, kernel_size=9, activation='softmax', padding='same')(x)

        model = models.Model(x_input, [output, z], name=self.name)

        #sparse_categorical_crossentropy -> labels' shape:[1,2,1]
        #categorical_crossentropy        -> labels' shape:[[0,1,0],[0,0,1],[0,1,0]]
        model.compile(
            optimizer=tf.keras.optimizers.Adadelta(lr=learning_rate),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy']
        )
        model.summary()
        return model

    def BatchGenerator(self, x_set, batch_size):
        return BatchGenerator(x_set, batch_size)


class BatchGenerator(Sequence):
    def __init__(self, x_set, batch_size):
        self.x= x_set
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
        batch_y = self.x[inds]
        return batch_x, batch_y

    def on_epoch_end(self):
        '''
        Called at the end of the epoch
        '''
        np.random.shuffle(self.indices)
