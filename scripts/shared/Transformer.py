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

        self.seq_length=512
        self.num_tokens=21 +3 #padding, bos, eos, mask, X aa
        self.token_embed_size=5

        self.d_model=32
        self.layers=2
        self.num_heads=self.d_model//16
        self.dff=self.d_model*4

        self.dropout_rate=0.01
        self.sub_ratio=0.15 #how many elements substitute: mask/random/none

    def positional_encoding(self, position, d_model):
        '''
        position: sequence length range: 0 to 511
        d_model: dimension of the embedding vector
        '''
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
            return pos * angle_rates

        angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(d_model)[np.newaxis, :],
                                d_model)
        
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)

    def custom_loss(self, y_true, y_pred):
        mask=tf.cast(tf.math.greater(y_true, 0), tf.float32)

        #encode labels
        y_true_encoded=tf.one_hot(tf.cast(y_true, tf.int32), self.num_tokens)

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

    def architecture(self, learning_rate=0.001):
        embedder=layers.Embedding(input_dim=self.num_tokens+1, output_dim=self.token_embed_size)
        x_input=layers.Input((self.seq_length), name='transformer_input')

        x=embedder(x_input)

        #project from emebdding to input transformer 5 -> d_model
        #avoiding sparse embeddings of size d_model (ALBERT trick) 
        x=layers.Dense(self.d_model)(x)

        #mask for attention (exluce padding from attention softmax)
        mask= tf.cast(tf.math.equal(x_input, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

        #Positional encoding
        x += self.positional_encoding(self.seq_length, self.d_model)

        for l in range(self.layers):
            x=EncoderLayer(self.d_model, self.num_heads, self.dff, self.dropout_rate)(x, mask=mask)

        #project to labels
        x=layers.Dense(self.num_tokens, name='transformer_output')(x)

        output=tf.math.softmax(x, axis=-1)

        transformer=models.Model(x_input, output, name='transformer')

        transformer.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss=self.custom_loss,
            metrics=[self.custom_accuracy]
        )

        transformer.summary()
        return transformer

    def BatchGenerator(self, x_set, batch_size):
        return BatchGenerator(x_set, batch_size, self.sub_ratio, self.num_tokens)

    def exportModel(self, model):
        for l in model.layers:
            print(l.name)


        skeleton=models.Model(
            inputs=model.get_layer('transformer_input').input,
            outputs=model.get_layer('encoder_layer').output#_'+str(self.layers-1)).output
        )

        skeleton.save(self.dir+'/model')


class BatchGenerator(Sequence):
    def __init__(self, x_set, batch_size, sub_ratio, num_tokens):
        self.x= x_set
        self.batch_size = batch_size
        self.indices = np.arange(self.x.shape[0])
        self.sub_ratio=sub_ratio
        self.mask_idx=num_tokens #value for mask token


    def __len__(self):
        '''
        Used by fit or fit_generator to set the number of steps per epoch
        '''
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, idx):
        b_start=idx * self.batch_size
        b_end=(idx + 1) * self.batch_size
        inds = self.indices[b_start:b_end]
        batch_x = self.x[np.sort(inds).tolist()]

        ##Get possible indexes to mask: no padding and BOS/EOS
        mask=batch_x!=0 #True is where it is possible to substitute

        #Get the coordinates in the tensor of available substitutes
        avail_mask_idxs=np.argwhere(mask) #(N, 2) N: number bases avail to substitute, 2: batch, position 

        #randomly pick the 15%
        pos_idxs=np.random.choice(np.arange(len(avail_mask_idxs)), int(len(avail_mask_idxs)*self.sub_ratio), replace=False) #get mask_idxs idxs to substitute

        #Get mask idxs
        substitue_idxs=avail_mask_idxs[pos_idxs]

        #set to 0 all 'labels' except the ones masked
        batch_y=batch_x*0
        batch_y[substitue_idxs[:, 0], substitue_idxs[:, 1]]=batch_x[substitue_idxs[:, 0], substitue_idxs[:, 1]]

        np.random.shuffle(substitue_idxs)# shuffle
        
        #among 15%: 20 random, 80 mask
        mask_idxs=substitue_idxs[:int(len(substitue_idxs)*0.8)]
        random_idxs=substitue_idxs[int(len(substitue_idxs)*0.8):]

        #apply mask to the input
        batch_x[mask_idxs[:, 0], mask_idxs[:, 1]]=self.mask_idx
        batch_x[random_idxs[:, 0], random_idxs[:, 1]]=np.random.randint(1, self.mask_idx, (len(random_idxs)))

        return batch_x, batch_y

    def on_epoch_end(self):
        '''
        Called at the end of the epoch
        '''
        np.random.shuffle(self.indices)

class MultiHeadAttention(tf.keras.layers.Layer):
    '''
    Multi-Head Attention:
        1. 
    '''
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead) 
        but it must be broadcastable for addition.
        
        Args:
            q: query shape == (..., seq_len_q, depth)
            k: key shape == (..., seq_len_k, depth)
            v: value shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable 
                to (..., seq_len_q, seq_len_k). Defaults to None.
            
        Returns:
            output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
            
        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    '''
    Encoder:
        1. Multi-head attention
        2. Dropout + Normalization
        3. FFC (2): (dff, d_model)
        4. Dropout + Normalization
    
    Input shape=output shape
    '''
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.dff=dff


        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = self.point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    #Required to save model
    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff})
        return config

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([
            layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
            layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

    def call(self, x, training, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, seq_length, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, seq_length, d_model)
        
        ffn_output = self.ffn(out1)  # (batch_size, seq_length, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, seq_length, d_model)
        
        return out2