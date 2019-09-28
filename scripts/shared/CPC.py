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

        self.sequence_length=512
        self.num_tokens=21
        self.token_embed_size=5
        self.code_size=128
        self.rnn_units=256
        self.num_predic_terms=8
        self.num_samples=6

    def BatchGenerator(self, x_set,batch_size):
        return BatchGenerator(x_set, batch_size)

    def buildEncoder(self):
        x_input=layers.Input((self.sequence_length, self.token_embed_size))
        
        x=x_input
        for num_kernels, _strides in zip([64, 64, 128, 128, 256], [2,1,2,1, 2]):
            shortcut=x
            x=layers.Conv1D(num_kernels, kernel_size=3, strides=_strides, activation='linear')(x)
            x=layers.BatchNormalization()(x)
            x=layers.LeakyReLU()(x)
            '''
            x=layers.Conv1D(num_kernels, kernel_size=3, strides=1, activation='linear', padding='same')(x)
            x=layers.BatchNormalization()(x)

            if (_strides!=1) or (shortcut.shape[-1]!=x.shape[-1]):
                shortcut=layers.Conv1D(num_kernels, kernel_size=1, strides=_strides, activation='linear', padding='same')(shortcut)
                shortcut=layers.BatchNormalization()(shortcut)

            x=layers.add([shortcut, x])
            x=layers.LeakyReLU()(x)
            '''
        output=layers.Conv1D(self.code_size, kernel_size=1, strides=1, activation='linear', name='encoder_embedding')(x)

        encoder_model = models.Model(x_input, output, name='encoder')

        return encoder_model

    def buildPredictorNetwork(self):
        #Define predictor network
        context_input=layers.Input((self.rnn_units))

        outputs = []
        for i in range(self.num_predic_terms):
            outputs.append(layers.Dense(units=self.code_size, activation="linear", name='z_t_{i}'.format(i=i))(context_input))

        def stack_outputs(x):
            import tensorflow as tf
            return tf.stack(x, axis=1)

        output=layers.Lambda(stack_outputs)(outputs)

        predictor_model = models.Model(context_input, output, name='predictor')

        return predictor_model

    def custom_loss(self, _, y_pred):
        labels=tf.expand_dims(tf.ones_like(y_pred[:, :, :, 0]), axis=-1)
        labels=tf.pad(labels, ((0,0), (0,0), (0,0), (0, self.num_samples-1)), "CONSTANT")

        losses=tf.keras.losses.categorical_crossentropy(
            labels,
            y_pred
        ) #batch, timesteps, predictions

        return tf.math.reduce_mean(losses)

    def custom_accuracy(self, _, y_pred):
        labels=tf.expand_dims(tf.ones_like(y_pred[:, :, :, 0]), axis=-1)
        labels=tf.pad(labels, ((0,0), (0,0), (0,0), (0, self.num_samples-1)), "CONSTANT")

        acc=tf.keras.metrics.categorical_accuracy(
            labels,
            y_pred
        ) #batch, timesteps, predictions

        return tf.math.reduce_mean(acc)

    def architecture(self, learning_rate=0.001):
        #Build model parts
        encoder_model=self.buildEncoder()
        embedder=layers.Embedding(input_dim=self.num_tokens, output_dim=self.token_embed_size)#, mask_zero=True)
        autoregressive_model=layers.LSTM(units=self.rnn_units, return_sequences=True, name='rnn')
        predictor_model=self.buildPredictorNetwork()

        ##1. Process Input Data
        x_input = layers.Input((self.sequence_length), name='encoder_input')
        x=embedder(x_input)

        x_encoded=encoder_model(x)#batch, timesteps, code_size
        rnn_output=autoregressive_model(x_encoded) #batch, timesteps, rnn_units

        #Predict next N code_size at each timestep
        preds=layers.TimeDistributed(predictor_model)(rnn_output) # batch, timesteps, num_preds, code_size
        
        #>>TARGET TRUE: [batch, timesteps, num_preds, code_size] (shifted by 1 timestep ahead)
        #Helper to gather true targets
        padded_x_encoded=tf.pad(x_encoded, ((0,0), (0, self.num_predic_terms), (0,0)), "CONSTANT")

        #gather timestep idxs to retrieve in the exact order: [2,3,4], [3,4,5] ..
        idxs=[]
        for i in range(1, preds.shape[1]+1):
            idxs.append(tf.range(i, i+self.num_predic_terms))
        idxs=tf.reshape(tf.stack(idxs), [-1])

        true_targets=tf.reshape(tf.gather(padded_x_encoded, idxs,axis=1), (-1, x_encoded.shape[1], self.num_predic_terms, self.code_size))
        true_targets=tf.expand_dims(true_targets, axis=-2)

        ##2. Process fake targets
        y_input = layers.Input((self.sequence_length), name='fake_targets_input')
        y_embedded=embedder(y_input)

        y_encoded=encoder_model(y_embedded)#batch, timesteps, code_size

        #>>TARGET FALSE: [batch, timesteps, num_preds, (num_samples-1) code_size]
        #y_fake comes with examples shuffled. So, sample num_preds*(num_samples-1) codes to create fake codes for given example
        idxs=[]
        for i in range(y_encoded.shape[1]):
            n_samples=self.num_predic_terms*(self.num_samples-1)
            idxs.append(tf.random.uniform(shape=[n_samples], minval=0, maxval=y_encoded.shape[1], dtype=tf.int32))
        idxs=tf.reshape(tf.stack(idxs), [-1])
        fake_targets=tf.reshape(tf.gather(y_encoded, idxs, axis=1), (-1, y_encoded.shape[1], self.num_predic_terms, (self.num_samples-1), self.code_size))

        expanded_preds=tf.expand_dims(preds, axis=-2)
        targets=tf.concat([true_targets, fake_targets], axis=-2)

        #Compute loss
        dot_product=tf.math.reduce_sum(expanded_preds*targets, axis=-1) #batch, timesteps, num_preds, samples

        #Each batch, timestep, pred is a vector of length 'sample' at which softmax is applied
        out=tf.math.softmax(dot_product)#batch, timesteps, num_preds, samples

        #Build model
        cpc = models.Model(inputs=[x_input, y_input], outputs=out, name=self.name)

        # Compile model
        cpc.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=self.custom_loss,
            metrics=[self.custom_accuracy]
        )

        #encoder_model.summary()
        #cpc_model.summary()

        return cpc

class BatchGenerator(Sequence):
    def __init__(self, x_set, batch_size):
        self.x = x_set
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
        batch_data = self.x[inds]

        np.random.shuffle(inds)

        target_batch=self.x[inds]

        return [batch_data, target_batch], batch_data

    def on_epoch_end(self):
        '''
        Called at the end of the epoch
        '''
        np.random.shuffle(self.indices)