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

        self.num_tokens=22 +1 #+1 for padding

        self.labels=8+1

    def custom_loss(self, y_true, y_pred):
        mask=tf.cast(tf.math.greater(y_true, 0), tf.float32)

        #encode labels
        y_true_encoded=tf.one_hot(tf.cast(y_true, tf.int32), self.labels)

        #compute loss
        Xentropy=tf.reduce_sum(-y_true_encoded*tf.math.log(y_pred+1e-6), axis=-1)
        #mask loss for padding positions
        Xentropy_masked=tf.math.multiply(Xentropy, mask)
        Xentropy_masked_batch_loss=tf.math.reduce_mean(Xentropy_masked)

        return Xentropy_masked_batch_loss

    def custom_accuracy(self, y_true, y_pred):
        #Generate mask
        mask=tf.math.greater(y_true, 0)

        #encode labels
        y_pred=tf.math.argmax(y_pred, axis=-1)

        #compute accuracy
        acc=tf.cast(tf.equal(tf.cast(y_true, tf.int64), y_pred), tf.float32)

        #apply mask which flats everything but it is okay
        acc=tf.boolean_mask(acc, mask)

        return tf.math.reduce_mean(acc)

    def architecture(self, b_model, learning_rate=0.00001):
        base_model=tf.keras.Model(
            inputs=b_model.get_layer('transformer_input').input,
            outputs=b_model.get_layer('transformer_output').output
        )

        base_model.trainable=False

        x_input=base_model.input
        x=base_model(x_input)

        output=layers.Dense(self.labels, activation='softmax')(x)

        model=tf.keras.Model(
            inputs=x_input,
            outputs=output)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=self.custom_loss,
            metrics=[self.custom_accuracy]
        )
        return model

    def BatchGenerator(self, x_set, y_set, batch_size):
        return BatchGenerator(x_set, y_set, batch_size)


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
