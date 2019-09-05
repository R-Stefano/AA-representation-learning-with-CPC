import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.metrics as metrics 

class Model():
    def __init__(self):
        self.name='CPC/'

    def prepareBatch(self, batch):
        return None

    def buildEncoder(self, window_size, encoding_length, code_size):
        encoder_model = models.Sequential(name='encoder')

        for num_kernels in [32,64,128]:
            encoder_model.add(layers.Conv1D(num_kernels, 3, activation='linear', input_shape=(window_size, encoding_length)))
            encoder_model.add(layers.BatchNormalization())
            encoder_model.add(layers.LeakyReLU())

        encoder_model.add(layers.Flatten())
        encoder_model.add(layers.Dense(units=256, activation='linear'))
        encoder_model.add(layers.BatchNormalization())
        encoder_model.add(layers.LeakyReLU())
        encoder_model.add(layers.Dense(units=code_size, activation='linear', name='encoder_embedding'))

        return encoder_model

    def buildPredictorNetwork(self, rnn_units, num_predic_terms, code_size):
        #Define predictor network
        context_input=layers.Input((rnn_units))

        outputs = []
        for i in range(num_predic_terms):
            outputs.append(layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context_input))

        def stack_outputs(x):
            import tensorflow as tf
            return tf.stack(x, axis=1)

        output=layers.Lambda(stack_outputs)(outputs)

        predictor_model = models.Model(context_input, output, name='predictor')

        return predictor_model

    def custom_xent(self, labels, preds):
        labels=tf.cast(labels, tf.float32) #from int8 to float32

        loss=tf.keras.losses.categorical_crossentropy(
            labels,
            preds,
            from_logits=False
        )
        mean_loss=tf.math.reduce_mean(loss)
        return mean_loss

    def custom_accuracy(self, y_true, y_pred):
        y_true=tf.cast(y_true, tf.float32) #from int8 to float32

        true_idxs=tf.math.argmax(y_true, axis=-1)
        pred_idxs=tf.math.argmax(y_pred, axis=-1)

        return tf.math.reduce_mean(tf.cast(tf.math.equal(true_idxs, pred_idxs), tf.float32))

    def architecture(self, sequence_length, num_predic_terms, num_samples, window_size, encoding_length, code_size, rnn_units, learning_rate):
        #Build model parts
        encoder_model=self.buildEncoder(window_size, encoding_length, code_size)
        autoregressive_model=layers.LSTM(units=rnn_units, return_sequences=True, name='rnn')
        predictor_model=self.buildPredictorNetwork(rnn_units, num_predic_terms, code_size)

        #encoder_model.summary()
        #predictor_model.summary()

        ##Process Input Data
        x_input = layers.Input((sequence_length, window_size, encoding_length), name='encoder_input')
        #1. Encoder (CNN)
        x_reshaped=tf.reshape(x_input, (-1, window_size, encoding_length))#batch*timesteps, window_size, encoding_length
        #x_encoded = layers.TimeDistributed(encoder_model)(x_input) #batch, timesteps, code_size
        x_encoded=encoder_model(x_reshaped)#batch*timesteps, code_size
        x_encoded=tf.reshape(x_encoded, (-1, sequence_length, code_size))#batch, timesteps, code_size
        #2. Autoregressor (RNN)
        rnn_output=autoregressive_model(x_encoded) #batch, timesteps, rnn_units
        rnn_output=tf.reshape(rnn_output, (-1, rnn_units))
        #3. Predictor (Dense): Predict next N embeddings at each timestep
        #preds = layers.TimeDistributed(predictor_model)(autoregressive_output) #batch, timesteps, num_preds, code_size
        preds=predictor_model(rnn_output) # batch*timesteps*num_preds, code_size

        ##Process next embeddings input (real & fake)
        y_input=layers.Input((sequence_length, num_predic_terms, num_samples, window_size, encoding_length), name='encoder_input_target')
        #1. Encode
        y_reshaped=tf.reshape(y_input, (-1, window_size, encoding_length)) #batch*timesteps*samples, vector_window, code_size
        #y_reshaped=tf.reshape(y_input, (-1, sequence_length*num_predic_terms*num_samples, window_size, encoding_length)) #batch, timesteps, vector_window, code_size
        y_encoded=encoder_model(y_reshaped) #batch*timesteps*num_preds*num_samples, code_size


        #Reshape preds
        pred_embeds=tf.reshape(preds, (-1, sequence_length, num_predic_terms, 1, code_size))
        #Reshape target embeds
        target_embeds=tf.reshape(y_encoded, (-1, sequence_length, num_predic_terms, num_samples, code_size)) #batch, timesteps, num_preds, num_samples, code_size

        #Compute loss
        dot_product=tf.math.reduce_sum(pred_embeds*target_embeds, axis=-1) #batch, timesteps, num_preds, samples

        #Each batch, timestep, pred is a vector of length 'sample' at which softmax is applied
        out=tf.math.softmax(dot_product)#batch, timesteps, num_preds, samples

        #Build model
        cpc_model = models.Model(inputs=[x_input, y_input], outputs=out, name='CPCModel')

        # Compile model
        cpc_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=self.custom_xent, #labels come as indexes, not as one hot vectors
            metrics=[self.custom_accuracy]
        )
        cpc_model.summary()

        return cpc_model