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

        self.num_tokens=3 #total number of elments in encoding voc 20 AA + 0 for padding
        self.token_embed_size=5 #length of tokens embedding vector
        self.sequence_length=256
        self.rnn_units=16
        self.attention_units=10

    def BatchGenerator(self, x_set,batch_size):
        return BatchGenerator(x_set, batch_size)

    def buildAttention(self, attention_units, encoder_outputs, encoder_hidden):
        encoder_hidden_time_axis=tf.expand_dims(encoder_hidden, axis=1)
        w1=layers.Dense(attention_units)
        w2=layers.Dense(attention_units)
        v=layers.Dense(1)
        score=v(tf.math.tanh(w1(encoder_hidden_time_axis) + w2(encoder_outputs)))

        attention_weights=tf.math.softmax(score, axis=1)

        context_vector = attention_weights * encoder_outputs
        context_vector = tf.math.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def architecture(self, learning_rate=0.1):
        #Encoder
        x_encoder = layers.Input((self.sequence_length),name='encoder_input')

        embedder=layers.Embedding(input_dim=self.num_tokens, output_dim=self.token_embed_size, mask_zero=True)

        x_encoder_embedded=embedder(x_encoder)

        encoder_outputs, encoder_out, encoder_hidden=layers.LSTM(units=self.rnn_units, return_sequences=True, return_state=True, name='encoder_rnn')(x_encoder_embedded)
        #encoder_outputs: [B, sequence_length, rnn_units] | encoder_hidden: [B, rnn_units]

        '''
        #Attention
        context_vector, attention_weights=self.buildAttention(self.attention_units, encoder_outputs, encoder_hidden)
        #context vector: [B, rnn_units] | attention_weights: [B, sequence_length, 1]
        '''

        x_decoder=layers.Input((None, ),name='decoder_input')
        #Decoder
        '''
        x_decoder_embedded=embedder(x_decoder)

        decoder_outputs, _, _=layers.LSTM(units=self.rnn_units, return_sequences=True, return_state=True, name='decoder_rnn')(x_decoder_embedded, initial_state=[encoder_out, encoder_hidden])

        outputs=layers.Dense(self.num_tokens, activation='softmax')(decoder_outputs)
        '''

        outputs=layers.Dense(self.num_tokens, activation='softmax')(encoder_outputs)

        
        #Build model
        #seq2seq_model = models.Model(inputs=[x_encoder, x_decoder], outputs=[x_encoder_embedded, x_decoder_embedded], name=self.name)
        seq2seq_model = models.Model(inputs=[x_encoder, x_decoder], outputs=outputs, name=self.name)

        seq2seq_model.summary()

        # Compile model
        seq2seq_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=['accuracy']
        )

        return seq2seq_model

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
        encoder_input = self.x[inds]
        decoder_input = self.x[inds]

        #shift by 1 for decoder to avoid see its output
        decoder_input=np.pad(decoder_input, ((0,0), (1,0)))[:, :-1] #pad at column 0. Remove last element

        return [encoder_input, decoder_input], encoder_input

    def on_epoch_end(self):
        '''
        Called at the end of the epoch
        '''
        np.random.shuffle(self.indices)