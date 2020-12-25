#  Implement multi head self attention as a Keras layer

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.supports_masking = True       
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output, weights
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads
        })
        return config

# Implement a Transformer block as a layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.supports_masking = True
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def compute_mask(self, inputs, mask=None):
        return mask
        
    def call(self, inputs, training, mask=None):
        attn_output, att_weights = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # return self.layernorm2(out1 + ffn_output)
        return (att_weights, self.layernorm2(out1 + ffn_output))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config
      
# Implement embedding layer
# Two seperate embedding layers, one for tokens, one for token index (positions).

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, embedding_matrix):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding_matrix = embedding_matrix
        self.token_emb = layers.Embedding(input_dim=vocab_size, 
                                          output_dim=embed_dim,
                                          weights=[embedding_matrix], 
                                          trainable = True,
                                          mask_zero=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim,
                                        mask_zero=True)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'embedding_matrix': self.embedding_matrix
        })
        return config

# Tuner
class transformerHyperModel(HyperModel):

    def __init__(self, embedding_matrix, EMBEDDING_DIM, vocab_size, MAX_SEQUENCE_LENGTH):
        self.embedding_matrix = embedding_matrix
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.vocab_size = vocab_size
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH

    def build(self, hp):
      inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,))
      embedding_layer = TokenAndPositionEmbedding(MAX_SEQUENCE_LENGTH, 
                                                  vocab_size, 
                                                  EMBEDDING_DIM, 
                                                  embedding_matrix)
      x = embedding_layer(inputs)
      transformer_block = TransformerBlock(embed_dim=EMBEDDING_DIM, 
                                           num_heads=1, ff_dim=128)
      attn1 , x = transformer_block(x)
      x = GlobalAveragePooling1D()(x)
      x = Dropout(0.5)(x)
      x = Dense(units=hp.Choice('units',values=[128, 256, 512],default=128),activation='relu',)(x)
      x = Dropout(0.5)(x)
      outputs = Dense(2, activation="softmax")(x)
      model = keras.Model(inputs=inputs, outputs=outputs)

      model.compile(
          optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2], default=1e-3)),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])
              
      return model

tuner = Hyperband(transformerHyperModel(embedding_matrix, EMBEDDING_DIM, vocab_size, MAX_SEQUENCE_LENGTH),
                  objective = 'val_accuracy', 
                  max_epochs = 10,
                  factor = 5,
                  directory = modelDir, 
                  project_name = 'transformer_tuner')  

def scheduler(epoch, lr):
  if epoch < 5:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

my_callbacks = [
    tf.keras.callbacks.LearningRateScheduler(scheduler),
    tf.keras.callbacks.EarlyStopping(patience=5)]

tuner.search(train_dataset, 
              epochs=20, 
              validation_data=test_dataset,
              class_weight=class_weight,
              callbacks=my_callbacks)